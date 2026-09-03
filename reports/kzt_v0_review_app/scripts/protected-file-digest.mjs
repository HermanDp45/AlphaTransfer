import { createHash } from "node:crypto";

const ownerHashPattern = /^(?:[A-Fa-f0-9]{64})?$/u;
const projectIdPattern = /^[A-Za-z0-9][A-Za-z0-9_-]*$/u;

function utf8Source(bytes, path) {
  const source = bytes.toString("utf8");
  if (!Buffer.from(source, "utf8").equals(bytes)) {
    throw new Error(`Protected Data app runtime file is not valid UTF-8: ${path}`);
  }
  return source;
}

function skipJavascriptTrivia(source, offset) {
  while (offset < source.length) {
    if (/\s/u.test(source[offset])) {
      offset += 1;
    } else if (source.startsWith("//", offset)) {
      const lineEnd = source.slice(offset + 2).search(/[\r\n\u2028\u2029]/u);
      offset = lineEnd < 0 ? source.length : offset + 2 + lineEnd;
    } else if (source.startsWith("/*", offset)) {
      const commentEnd = source.indexOf("*/", offset + 2);
      if (commentEnd < 0) return -1;
      offset = commentEnd + 2;
    } else {
      break;
    }
  }
  return offset;
}

function normalizeOwner(bytes, path) {
  const source = utf8Source(bytes, path);
  const values = [];
  let offset = 0;
  while (skipJavascriptTrivia(source, offset) !== source.length) {
    const exportStart = skipJavascriptTrivia(source, offset);
    const prefix = /^export const (dataAppOwner(?:UserId|Email)Sha256) = "/u.exec(source.slice(exportStart));
    if (exportStart < 0 || !prefix) {
      throw new Error("The protected Data app owner module must contain exactly its expected owner exports.");
    }
    const start = exportStart + prefix[0].length;
    const end = source.indexOf('"', start);
    if (end < 0 || !source.startsWith('";', end)) {
      throw new Error("The protected Data app owner module must contain exactly its expected owner exports.");
    }
    const value = source.slice(start, end);
    if (!ownerHashPattern.test(value)) {
      throw new Error("The protected Data app owner seed must be empty or a SHA-256 hexadecimal string.");
    }
    values.push({ name: prefix[1], start, end, value });
    offset = end + 2;
  }
  // Recognize old static seed modules only to preserve/migrate existing artifacts.
  const names = values.map(({ name }) => name).join(",");
  if (!["dataAppOwnerEmailSha256", "dataAppOwnerUserIdSha256", "dataAppOwnerUserIdSha256,dataAppOwnerEmailSha256"].includes(names)) {
    throw new Error("The protected Data app owner module must contain exactly its expected owner exports.");
  }
  if (values.filter(({ value }) => value).length > 1) {
    throw new Error("The protected Data app user-ID and email owner seeds must be mutually exclusive.");
  }
  // Publication may change only the seed values; syntax and other bytes stay protected.
  const normalized = [];
  offset = 0;
  for (const { start, end } of values) {
    normalized.push(source.slice(offset, start));
    offset = end;
  }
  normalized.push(source.slice(offset));
  return Buffer.from(normalized.join(""), "utf8");
}

function normalizeHosting(bytes, path) {
  const source = utf8Source(bytes, path);
  let hosting;
  try {
    hosting = JSON.parse(source);
  } catch {
    throw new Error("The protected Data app hosting metadata is not valid JSON.");
  }
  if (!hosting || typeof hosting !== "object" || Array.isArray(hosting)) {
    throw new Error("The protected Data app hosting metadata must be an object.");
  }
  const hasProjectId = Object.hasOwn(hosting, "project_id");
  if (hasProjectId && (typeof hosting.project_id !== "string" || !projectIdPattern.test(hosting.project_id))) {
    throw new Error("The protected Data app hosting metadata has an invalid project_id.");
  }
  const keys = Object.keys(hosting);
  if (
    !hasProjectId ||
    hosting.d1 !== "DB" ||
    hosting.r2 !== null ||
    keys.length !== 3 ||
    !keys.every((key) => ["d1", "r2", "project_id"].includes(key)) ||
    source !== `${JSON.stringify(hosting, null, 2)}\n`
  ) {
    return bytes;
  }
  // Match only the ordinary Data/Sites writer output. Removing this one field
  // preserves the remaining keys and their order; custom bindings, extra keys,
  // duplicate keys, and other byte changes still require scoped authorization.
  const remaining = { ...hosting };
  delete remaining.project_id;
  return Buffer.from(`${JSON.stringify(remaining, null, 2)}\n`, "utf8");
}

/** Normalize only the two publication-owned fields of a protected relative path. */
export function normalizeProtectedFile(path, contents) {
  const bytes = Buffer.isBuffer(contents) ? contents : Buffer.from(contents);
  if (path === "src/data-app-owner.js") return normalizeOwner(bytes, path);
  if (path === ".openai/hosting.json") return normalizeHosting(bytes, path);
  return bytes;
}

export function digestProtectedFile(path, contents) {
  return createHash("sha256").update(normalizeProtectedFile(path, contents)).digest("hex");
}
