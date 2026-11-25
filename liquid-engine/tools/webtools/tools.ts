// web_search / web_fetch as liquid-server built-in tools, on the code vendored from the pi extensions
// (providers.ts, fetch.ts) and the pdf-reader skill (pdf). Names, descriptions, parameters and result text match the pi tools.
// The wire format is llama-server's server tools (GET/POST /tools, the same as get_info):
//
//   node tools.ts list                      -> [{display_name, tool, type: "server", permissions, uses_cwd, definition}]
//   node tools.ts call <name> < params.json -> {"plain_text_response": text} | {"error": message}
import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { searchWithFallback, type Recency, type SearchResult } from "./providers.ts";
import { fetchAndExtract } from "./fetch.ts";

const SNIPPET_CHARS = 500;

function normalizeDomain(value: string): string {
	let domain = value.trim().replace(/^site:/i, "");
	try {
		domain = new URL(/^[a-z]+:\/\//i.test(domain) ? domain : `https://${domain}`).hostname;
	} catch {}
	return domain.replace(/^www\./, "");
}

// Providers return ISO timestamps or text like "Nov 12, 2026"; show YYYY-MM-DD either way
function isoDate(raw: string): string {
	if (/^\d{4}-\d{2}-\d{2}/.test(raw)) return raw.slice(0, 10);
	const d = new Date(`${raw} UTC`);
	return Number.isNaN(d.getTime()) ? raw : d.toISOString().slice(0, 10);
}

function formatResults(provider: string, results: SearchResult[]): string {
	const lines = [`Provider: ${provider} (${results.length} results)`];
	results.forEach((r, i) => {
		let snippet = r.snippet.replace(/\s+/g, " ").trim();
		if (snippet.length > SNIPPET_CHARS) snippet = `${snippet.slice(0, SNIPPET_CHARS)}…`;
		const date = r.date ? ` (${isoDate(r.date)})` : "";
		lines.push(`\n${i + 1}. ${r.title}${date}\n   ${r.url}\n   ${snippet}`);
	});
	return lines.join("\n");
}

const TOOLS = [
	{
		name: "web_search",
		display_name: "Web Search",
		description:
			"Search the web. Returns title, URL, date and snippet per result. " +
			"Tries TinyFish, LangSearch, Exa, then Wikipedia (keyless), falling back automatically when a provider fails. " +
			"Use one web_search call per search angle; rephrase rather than repeating the same query. " +
			"Then use web_fetch on the one or two most relevant URLs for full content.",
		parameters: {
			type: "object",
			properties: {
				query: { type: "string", description: "Search query in plain words" },
				count: { type: "number", description: "Number of results (default 5)", minimum: 1, maximum: 10 },
				site: { type: "string", description: "Only return results from this domain, e.g. github.com" },
				exclude_sites: { type: "array", items: { type: "string" }, description: 'Domains to leave out, e.g. ["pinterest.com"]' },
				recency: { type: "string", enum: ["day", "week", "month", "year"], description: "Only results published within this period" },
			},
			required: ["query"],
		},
	},
	{
		name: "web_fetch",
		display_name: "Web Fetch",
		description:
			"Fetch a web page and extract readable content as clean markdown. Uses the TinyFish Fetch API first " +
			"(renders JS-heavy pages and PDFs), then Readability + Turndown locally, then Jina Reader. Handles PDFs and plain text.",
		parameters: {
			type: "object",
			properties: { url: { type: "string", description: "URL to fetch" } },
			required: ["url"],
		},
	},
];

// pdf: the pi pdf-reader skill's `pdf` command (vendored as ./pdf; wraps liteparse, caches per file, limits output)
const PDF_SCRIPT = path.join(path.dirname(fileURLToPath(import.meta.url)), "pdf");
const PDF_TIMEOUT_MS = 600_000;
const PDF_DOWNLOADS = path.join(os.tmpdir(), "liquid-pdf-downloads");

function findOnPath(bin: string, extra: string[] = []): string | null {
	for (const d of [...(process.env.PATH ?? "").split(path.delimiter), ...extra]) {
		if (!d) continue;
		try {
			fs.accessSync(path.join(d, bin), fs.constants.X_OK);
			return path.join(d, bin);
		} catch {}
	}
	return null;
}

const LITEPARSE_BIN = process.env.LITEPARSE_BIN || findOnPath("liteparse", [path.join(os.homedir(), ".local", "bin")]);
const PDF_AVAILABLE = !!LITEPARSE_BIN && !!findOnPath("jq") && fs.existsSync(PDF_SCRIPT);

const PDF_TOOL = {
	name: "pdf",
	display_name: "PDF Reader",
	uses_cwd: true,
	description:
		"Read a PDF or document (DOCX/PPTX/XLSX, images) page by page without flooding context. " +
		"file is a local path (relative to the working directory) or an http(s) URL. Commands: " +
		"map = step 1, always: one row per page (chars, OCR needed, columns, tables, figures); " +
		"ocr = OCR the pages map marked OCR (or pages like \"4,9-12\"); " +
		"outline = headings with page numbers (depth for deeper levels); " +
		"find = case-insensitive regex search, one line per match \"p12: ...\"; " +
		"page = read one page (long pages say which line to continue from). " +
		"Work: map, then outline/find to locate, then read only those pages. Cite page numbers; say when text came from OCR. " +
		"Hints in the output written as `pdf page FILE N LINE` mean: call this tool with command \"page\", page N, line LINE.",
	parameters: {
		type: "object",
		properties: {
			command: { type: "string", enum: ["map", "ocr", "outline", "find", "page"], description: "What to do" },
			file: { type: "string", description: "Local path or http(s) URL of the document" },
			page: { type: "number", description: "page: page number to read" },
			line: { type: "number", description: "page: line to continue from (default 1)" },
			regex: { type: "string", description: "find: regular expression" },
			pages: { type: "string", description: 'ocr: pages to OCR, e.g. "4,9-12" (default: the pages map marked OCR)' },
			depth: { type: "number", description: "outline: heading depth 1-6 (default: the deepest that fits)" },
		},
		required: ["command", "file"],
	},
};

const ALL_TOOLS: Array<(typeof TOOLS)[number] & { uses_cwd?: boolean }> = PDF_AVAILABLE ? [...TOOLS, PDF_TOOL] : TOOLS;

// llama-server's server_tool::to_json() shape
const SERVER_TOOLS = ALL_TOOLS.map(({ name, display_name, description, parameters, uses_cwd }) => ({
	display_name,
	tool: name,
	type: "server",
	permissions: { write: false },
	uses_cwd: uses_cwd ?? false,
	definition: { type: "function", function: { name, description, parameters } },
}));

const DOC_TYPES: Record<string, string> = {
	"application/pdf": "pdf",
	"application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
	"application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
	"application/msword": "doc",
	"application/vnd.ms-excel": "xls",
	"application/vnd.ms-powerpoint": "ppt",
	"application/vnd.oasis.opendocument.text": "odt",
	"application/vnd.oasis.opendocument.spreadsheet": "ods",
	"application/vnd.oasis.opendocument.presentation": "odp",
	"image/png": "png",
	"image/jpeg": "jpg",
	"image/webp": "webp",
	"image/tiff": "tiff",
};
const KNOWN_EXTS = new Set([...Object.values(DOC_TYPES), "jpeg", "tif"]);

// A URL is downloaded once to a path derived from it, so the pdf script's per-file cache works across calls.
async function downloadDocument(url: string): Promise<string> {
	const key = createHash("sha256").update(url).digest("hex").slice(0, 16);
	fs.mkdirSync(PDF_DOWNLOADS, { recursive: true });
	const existing = fs.readdirSync(PDF_DOWNLOADS).find((f) => f.startsWith(`${key}.`));
	if (existing) return path.join(PDF_DOWNLOADS, existing);
	const res = await fetch(url, { redirect: "follow", signal: AbortSignal.timeout(120_000) });
	if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
	const urlExt = path.extname(new URL(res.url || url).pathname).slice(1).toLowerCase();
	const ctype = (res.headers.get("content-type") ?? "").split(";")[0].trim().toLowerCase();
	const ext = DOC_TYPES[ctype] ?? (KNOWN_EXTS.has(urlExt) ? urlExt : "pdf");
	const file = path.join(PDF_DOWNLOADS, `${key}.${ext}`);
	fs.writeFileSync(`${file}.part`, Buffer.from(await res.arrayBuffer()));
	fs.renameSync(`${file}.part`, file);
	return file;
}

interface PdfParams { command: string; file: string; page?: number; line?: number; regex?: string; pages?: string; depth?: number; cwd?: string }

async function pdfTool(p: PdfParams): Promise<string> {
	const target = p.file?.trim();
	if (!target) throw new Error("'file' is required.");
	let file: string;
	if (/^https?:\/\//i.test(target)) file = await downloadDocument(target);
	else if (target.startsWith("~/")) file = path.join(os.homedir(), target.slice(2));
	else file = path.resolve(p.cwd || process.cwd(), target);
	const args = [p.command, file];
	const num = (v: unknown, what: string) => {
		if (typeof v !== "number" || !Number.isFinite(v)) throw new Error(`'${what}' is required for ${p.command}`);
		return String(Math.trunc(v));
	};
	switch (p.command) {
		case "map":
			break;
		case "ocr":
			if (p.pages) args.push(String(p.pages));
			break;
		case "outline":
			if (p.depth !== undefined) args.push(num(p.depth, "depth"));
			break;
		case "find":
			if (!p.regex) throw new Error("'regex' is required for find");
			args.push(p.regex);
			break;
		case "page":
			args.push(num(p.page, "page"));
			if (p.line !== undefined) args.push(num(p.line, "line"));
			break;
		default:
			throw new Error(`unknown command "${p.command}" (map, ocr, outline, find, page)`);
	}
	const env = { ...process.env, PATH: [path.dirname(LITEPARSE_BIN!), process.env.PATH].join(path.delimiter) };
	return new Promise((resolve, reject) => {
		execFile(PDF_SCRIPT, args, { env, maxBuffer: 64 << 20, timeout: PDF_TIMEOUT_MS }, (err, stdout, stderr) => {
			const out = stdout.trim();
			if (err) reject(new Error((stderr.trim() || err.message) + (out ? `\n${out}` : "")));
			else resolve(out || stderr.trim());
		});
	});
}

interface SearchParams { query: string; count?: number; site?: string; exclude_sites?: string[]; recency?: Recency }

async function webSearch(params: SearchParams, signal?: AbortSignal): Promise<string> {
	const query = params.query?.trim();
	if (!query) throw new Error("'query' is required.");
	const outcome = await searchWithFallback({
		query,
		count: Math.min(Math.max(Math.round(params.count ?? 5), 1), 10),
		countRequested: params.count !== undefined,
		includeDomains: params.site ? [normalizeDomain(params.site)] : undefined,
		excludeDomains: params.exclude_sites?.map(normalizeDomain).filter(Boolean),
		recency: params.recency,
		signal,
	});
	return formatResults(outcome.provider, outcome.results);
}

async function webFetch(params: { url: string }, signal?: AbortSignal): Promise<string> {
	const result = await fetchAndExtract(params.url, signal);
	if (result.error) throw new Error(`${params.url}: ${result.error}`);
	const header = result.title ? `# ${result.title}\n\nSource: ${result.url}\n\n---\n\n` : "";
	return header + result.content;
}

async function readStdin(): Promise<string> {
	const chunks: Buffer[] = [];
	for await (const c of process.stdin) chunks.push(c as Buffer);
	return Buffer.concat(chunks).toString("utf8");
}

async function main() {
	const [cmd, name] = process.argv.slice(2);
	if (cmd === "list") {
		process.stdout.write(JSON.stringify(SERVER_TOOLS));
		return;
	}
	if (cmd !== "call" || !name) {
		process.stderr.write("usage: tools.ts list | call <name> < params.json\n");
		process.exit(2);
	}
	let out: { plain_text_response: string } | { error: string };
	try {
		const raw = (await readStdin()).trim();
		const args = raw ? JSON.parse(raw) : {};
		if (name === "web_search") out = { plain_text_response: await webSearch(args) };
		else if (name === "web_fetch") out = { plain_text_response: await webFetch(args) };
		else if (name === "pdf" && PDF_AVAILABLE) out = { plain_text_response: await pdfTool(args) };
		else throw new Error(`unknown tool "${name}"`);
	} catch (err) {
		out = { error: err instanceof Error ? err.message : String(err) };
	}
	process.stdout.write(JSON.stringify(out));
}

main();
