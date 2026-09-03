const snapshotId = "current";
const rowsPerStatement = 30;

const schemas = [
  `CREATE TABLE IF NOT EXISTS data_app_snapshots (
    id TEXT PRIMARY KEY,
    metadata_json TEXT NOT NULL,
    seed_sha256 TEXT NOT NULL
  )`,
  `CREATE TABLE IF NOT EXISTS data_app_queries (
    id TEXT PRIMARY KEY,
    position INTEGER NOT NULL,
    query_json TEXT NOT NULL
  )`,
  `CREATE TABLE IF NOT EXISTS data_app_query_rows (
    query_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    row_json TEXT NOT NULL,
    PRIMARY KEY (query_id, position)
  )`,
];

async function fingerprint(snapshot) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(JSON.stringify(snapshot)));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function insertRows(database, queryId, rows) {
  const statements = [];
  for (let offset = 0; offset < rows.length; offset += rowsPerStatement) {
    const chunk = rows.slice(offset, offset + rowsPerStatement);
    const placeholders = chunk.map(() => "(?, ?, ?)").join(", ");
    const values = chunk.flatMap((row, index) => [queryId, offset + index, JSON.stringify(row)]);
    statements.push(database.prepare(
      `INSERT INTO data_app_query_rows (query_id, position, row_json) VALUES ${placeholders}`,
    ).bind(...values));
  }
  return statements;
}

async function initializeSnapshot(database, snapshot) {
  if (typeof database?.prepare !== "function" || typeof database.batch !== "function") {
    throw new Error("The Sites D1 database is unavailable.");
  }

  await database.batch(schemas.map((schema) => database.prepare(schema)));
  const seedHash = await fingerprint(snapshot);
  const previous = await database.prepare(
    "SELECT seed_sha256 FROM data_app_snapshots WHERE id = ?",
  ).bind(snapshotId).first();
  if (previous?.seed_sha256 === seedHash) return;

  const { queries = {}, ...metadata } = snapshot;
  const statements = [
    database.prepare("DELETE FROM data_app_query_rows"),
    database.prepare("DELETE FROM data_app_queries"),
    database.prepare(
      "INSERT INTO data_app_snapshots (id, metadata_json, seed_sha256) VALUES (?, ?, ?) "
        + "ON CONFLICT(id) DO UPDATE SET metadata_json = excluded.metadata_json, seed_sha256 = excluded.seed_sha256",
    ).bind(snapshotId, JSON.stringify(metadata), seedHash),
  ];

  for (const [position, [queryId, query]] of Object.entries(queries).entries()) {
    const { rows = [], ...definition } = query;
    statements.push(database.prepare(
      "INSERT INTO data_app_queries (id, position, query_json) VALUES (?, ?, ?)",
    ).bind(queryId, position, JSON.stringify(definition)));
    statements.push(...insertRows(database, queryId, rows));
  }

  await database.batch(statements);
}

export async function storedSnapshot(database, seedSnapshot) {
  await initializeSnapshot(database, seedSnapshot);
  const [snapshot, queryResult, rowResult] = await Promise.all([
    database.prepare("SELECT metadata_json FROM data_app_snapshots WHERE id = ?").bind(snapshotId).first(),
    database.prepare("SELECT id, query_json FROM data_app_queries ORDER BY position").all(),
    database.prepare(
      "SELECT query_id, row_json FROM data_app_query_rows ORDER BY query_id, position",
    ).all(),
  ]);

  const queries = Object.fromEntries(queryResult.results.map(({ id, query_json }) => [
    id,
    { ...JSON.parse(query_json), rows: [] },
  ]));
  for (const { query_id, row_json } of rowResult.results) queries[query_id].rows.push(JSON.parse(row_json));
  return { ...JSON.parse(snapshot.metadata_json), queries };
}

export async function updateStoredQuery(database, queryId, rows, generatedAt) {
  await database.batch([
    database.prepare("DELETE FROM data_app_query_rows WHERE query_id = ?").bind(queryId),
    ...insertRows(database, queryId, rows),
    database.prepare(
      "UPDATE data_app_snapshots SET metadata_json = json_set(metadata_json, '$.generatedAt', ?) WHERE id = ?",
    ).bind(generatedAt, snapshotId),
  ]);
}
