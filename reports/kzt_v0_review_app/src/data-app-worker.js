import { validatePresentation } from "./presentation-state.js";
import { storedSnapshot, updateStoredQuery } from "./snapshot-storage.js";
import { normalizeOwnerEmail } from "./owner-email.js";

export { validatePresentation };

const id = "current";
const presentationSchemaSql = `
  CREATE TABLE IF NOT EXISTS data_app_presentation_v1 (
    id TEXT PRIMARY KEY,
    presentation_json TEXT NOT NULL,
    revision INTEGER NOT NULL,
    updated_at TEXT NOT NULL
  )
`;

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "private, no-store",
    },
  });
}

function componentPermalinkPath(pathname) {
  const match = /^\/_data\/(charts|components)\/([^/]+)(\/detail)?$/u.exec(pathname);
  if (!match || (match[1] === "components" && match[3])) return false;

  try {
    const componentId = decodeURIComponent(match[2]);
    return (
      componentId.length <= 200 &&
      Boolean(componentId.trim()) &&
      componentId !== "." &&
      componentId !== ".." &&
      !/[\\/\0]/u.test(componentId)
    );
  } catch {
    return false;
  }
}

function authenticatedViewerEmail(request) {
  return normalizeOwnerEmail(request.headers.get("oai-authenticated-user-email"));
}

async function sha256(value) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function isOwnerHash(value) {
  return typeof value === "string" && /^[a-f\d]{64}$/u.test(value);
}

function htmlWithSitesProject(dataAppHtml, projectId) {
  if (!projectId) return dataAppHtml;
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/u.test(projectId)) {
    throw new Error("The Sites project ID is invalid.");
  }
  // Inspect head markup, not matching strings inside scripts, styles, or comments.
  const tags = /<!--[\s\S]*?-->|<(script|style|title|textarea)\b(?:"[^"]*"|'[^']*'|[^'">])*>[\s\S]*?<\/\1\s*>|<(\/?)([a-z][a-z\d:-]*)\b((?:"[^"]*"|'[^']*'|[^'">])*)>/giu;
  let head;
  for (const tag of dataAppHtml.matchAll(tags)) {
    const closing = tag[2];
    const name = tag[3]?.toLowerCase();
    if (!head) {
      if (name === "head" && !closing) head = tag;
      continue;
    }
    if (name === "head" && closing) break;
    if (name !== "meta" || closing) continue;
    const attributes = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/gu;
    for (const attribute of tag[4].matchAll(attributes)) {
      if (
        attribute[1].toLowerCase() === "name" &&
        (attribute[2] ?? attribute[3] ?? attribute[4] ?? "").toLowerCase() === "data-app-sites-project"
      ) {
        throw new Error("The reviewed Data app HTML must not define its Sites project identity.");
      }
    }
  }
  if (!head) throw new Error("The reviewed Data app HTML must contain a head element.");
  const marker = `<meta name="data-app-sites-project" content="${projectId}">`;
  return `${dataAppHtml.slice(0, head.index + head[0].length)}${marker}${dataAppHtml.slice(head.index + head[0].length)}`;
}

export function createDataAppWorker({
  html: dataAppHtml,
  projectId = "",
  seedSnapshot,
  ownerEmailSha256 = "",
  initialPresentation = {},
}) {
  const servedHtml = htmlWithSitesProject(dataAppHtml, projectId);
  const presentationSeed = { ...validatePresentation(initialPresentation) };
  // A reviewed local seed cannot fabricate a creator-verified badge.
  delete presentationSeed.verification;

  async function viewerCanEdit(request) {
    if (!isOwnerHash(ownerEmailSha256)) return false;
    // Sites replaces this header with the authenticated visitor's email.
    const email = authenticatedViewerEmail(request);
    return email !== null && (await sha256(email)) === ownerEmailSha256;
  }

  async function storedPresentation(database) {
    if (!database?.prepare) throw new Error("The Sites D1 database is unavailable.");
    await database.prepare(presentationSchemaSql).run();
    await database
      .prepare(
        "INSERT INTO data_app_presentation_v1 (id, presentation_json, revision, updated_at) " +
          "VALUES (?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
      )
      .bind(id, JSON.stringify(presentationSeed), 0, new Date().toISOString())
      .run();
    const row = await database
      .prepare("SELECT presentation_json, revision, updated_at FROM data_app_presentation_v1 WHERE id = ?")
      .bind(id)
      .first();
    return { presentation: JSON.parse(row.presentation_json), revision: row.revision, updatedAt: row.updated_at };
  }

  return {
    async fetch(request, environment) {
      const { pathname } = new URL(request.url);
      const isComponentPermalink =
        (request.method === "GET" || request.method === "HEAD") && componentPermalinkPath(pathname);
      if (pathname === "/" || pathname === "/index.html" || isComponentPermalink) {
        return new Response(isComponentPermalink && request.method === "HEAD" ? null : servedHtml, {
          headers: { "content-type": "text/html; charset=utf-8" },
        });
      }

      if (pathname === "/api/snapshot" && request.method === "GET") {
        return json(await storedSnapshot(environment.DB, seedSnapshot));
      }

      if (pathname === "/api/presentation" && request.method === "GET") {
        return json({ ...(await storedPresentation(environment.DB)), canEdit: await viewerCanEdit(request) });
      }

      if (pathname === "/api/presentation" && request.method === "PUT") {
        if (!(await viewerCanEdit(request))) {
          return json({ error: "Only the current Site owner can edit the Data app presentation." }, 403);
        }
        let body;
        let presentation;
        try {
          body = await request.json();
          presentation = validatePresentation(body.presentation);
        } catch (error) {
          return json({ error: error instanceof Error ? error.message : "Presentation is invalid." }, 400);
        }
        if (!Number.isSafeInteger(body.revision) || body.revision < 0) {
          return json({ error: "A valid presentation revision is required." }, 400);
        }
        if (
          body.verificationAction !== undefined &&
          body.verificationAction !== "verify" &&
          body.verificationAction !== "remove"
        ) {
          return json({ error: "Dashboard verification action must be verify or remove." }, 400);
        }
        const current = await storedPresentation(environment.DB);
        if (current.revision !== body.revision) return json(current, 409);
        const updatedAt = new Date().toISOString();
        if (body.verificationAction === "remove") {
          delete presentation.verification;
        } else if (body.verificationAction === "verify") {
          presentation.verification = current.presentation.verification ?? {
            verifiedBy: authenticatedViewerEmail(request), verifiedAt: updatedAt,
          };
        } else if (presentation.verification) {
          if (!current.presentation.verification) {
            return json({ error: "Dashboard verification requires an explicit server-side verification action." }, 400);
          }
          presentation.verification = current.presentation.verification;
        }
        const result = await environment.DB.prepare(
          "UPDATE data_app_presentation_v1 SET presentation_json = ?, revision = revision + 1, " +
            "updated_at = ? WHERE id = ? AND revision = ?",
        )
          .bind(JSON.stringify(presentation), updatedAt, id, body.revision)
          .run();
        if (!result.meta?.changes) return json(await storedPresentation(environment.DB), 409);
        return json({ presentation, revision: body.revision + 1, updatedAt });
      }

      if (pathname.startsWith("/api/queries/") && request.method === "PUT") {
        if (!(await viewerCanEdit(request))) {
          return json({ error: "Only the current Site owner can update reviewed Data app data." }, 403);
        }

        const queryId = decodeURIComponent(pathname.slice("/api/queries/".length));
        const snapshot = await storedSnapshot(environment.DB, seedSnapshot);
        if (!Object.hasOwn(snapshot.queries, queryId)) {
          return json({ error: "Data app query was not found." }, 404);
        }

        const { rows } = await request.json();
        if (
          !Array.isArray(rows) ||
          rows.length > 10_000 ||
          rows.some((row) => row === null || typeof row !== "object" || Array.isArray(row))
        ) {
          return json({ error: "Data app query rows are invalid." }, 400);
        }

        const generatedAt = new Date().toISOString();
        await updateStoredQuery(environment.DB, queryId, rows, generatedAt);
        return json({ queryId, rows, generatedAt });
      }

      return new Response("Not found", { status: 404 });
    },
  };
}
