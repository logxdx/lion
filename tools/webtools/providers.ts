// From the pi web-search extension (~/.pi/agent/extensions/web-search/providers.ts); the keyless Python ddgs
// fallback is replaced by the Wikipedia search API (see "Keyless fallback").
import { execFile } from "node:child_process";

// ── Unified interface ────────────────────────────────────────────────
//
// Every provider takes the same request and returns the same result shape.
// To add a provider, implement SearchProvider and add it to PROVIDERS.

export type Recency = "day" | "week" | "month" | "year";

export interface SearchRequest {
	query: string;
	/** Results to return to the agent. */
	count: number;
	/** True when the caller asked for `count`; false when it's the default. */
	countRequested?: boolean;
	includeDomains?: string[];
	excludeDomains?: string[];
	recency?: Recency;
	signal?: AbortSignal;
}

export interface SearchResult {
	title: string;
	url: string;
	snippet: string;
	date?: string;
}

export interface SearchProvider {
	name: string;
	/** Name of the API key (keyring `key` attribute / env var); undefined for keyless providers. */
	keyEnv?: string;
	search(req: SearchRequest, key: string | undefined): Promise<SearchResult[]>;
}

export class ProviderError extends Error {}

const TIMEOUT_MS = 20000;

async function request(url: string, init: RequestInit, signal?: AbortSignal): Promise<any> {
	let res: Response;
	try {
		res = await fetch(url, {
			...init,
			signal: AbortSignal.any([AbortSignal.timeout(TIMEOUT_MS), ...(signal ? [signal] : [])]),
		});
	} catch (err) {
		throw new ProviderError(err instanceof Error ? err.message : String(err));
	}
	if (!res.ok) {
		const body = (await res.text().catch(() => "")).replace(/\s+/g, " ").slice(0, 200);
		throw new ProviderError(`HTTP ${res.status} ${body}`);
	}
	return res.json();
}

const RECENCY_DAYS: Record<Recency, number> = { day: 1, week: 7, month: 30, year: 365 };

// ── Providers ────────────────────────────────────────────────────────

/** https://docs.tinyfish.ai/search-api/reference — free; daily allowance, 30 req/min. */
const tinyfish: SearchProvider = {
	name: "tinyfish",
	keyEnv: "TINYFISH_API_KEY",
	async search(req, key) {
		const url = new URL("https://api.search.tinyfish.ai");
		url.searchParams.set("query", req.query);
		if (req.includeDomains?.length) url.searchParams.set("include_domains", req.includeDomains.join(","));
		if (req.excludeDomains?.length) url.searchParams.set("exclude_domains", req.excludeDomains.join(","));
		if (req.recency) url.searchParams.set("recency_minutes", String(RECENCY_DAYS[req.recency] * 1440));
		const data = await request(url.toString(), { headers: { "X-API-Key": key! } }, req.signal);
		// No result-count parameter: the API returns a page of results
		return ((data.results ?? []) as any[]).slice(0, req.count).map((r) => ({
			title: r.title ?? "",
			url: r.url ?? "",
			snippet: r.snippet ?? "",
			date: r.date ?? undefined,
		}));
	},
};

/** https://langsearch.com/install/langsearch/SKILL.md — free. */
const langsearch: SearchProvider = {
	name: "langsearch",
	keyEnv: "LANGSEARCH_API_KEY",
	async search(req, key) {
		const freshness = { day: "oneDay", week: "oneWeek", month: "oneMonth", year: "oneYear" } as const;
		const data = await request(
			"https://api.langsearch.com/v1/web-search",
			{
				method: "POST",
				headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
				body: JSON.stringify({
					query: req.query,
					count: req.count,
					...(req.includeDomains?.length && { includeDomains: req.includeDomains }),
					...(req.excludeDomains?.length && { excludeDomains: req.excludeDomains }),
					...(req.recency && { freshness: freshness[req.recency] }),
				}),
			},
			req.signal,
		);
		return ((data?.data?.webPages?.value ?? []) as any[]).map((p) => ({
			title: p.name ?? "",
			url: p.url ?? "",
			snippet: p.snippet ?? p.text ?? "",
			date: p.datePublished ?? undefined,
		}));
	},
};

/** Exa /search per the build-with-exa skill: query + highlights, `instant` for tool latency. */
const exa: SearchProvider = {
	name: "exa",
	keyEnv: "EXA_API_KEY",
	async search(req, key) {
		const since = req.recency
			? new Date(Date.now() - RECENCY_DAYS[req.recency] * 86400000).toISOString()
			: undefined;
		const data = await request(
			"https://api.exa.ai/search",
			{
				method: "POST",
				headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
				body: JSON.stringify({
					query: req.query,
					type: "instant",
					// Exa's guidance: omit numResults (default 10) unless a count was asked for
					...(req.countRequested && { numResults: req.count }),
					contents: { highlights: true },
					...(req.includeDomains?.length && { includeDomains: req.includeDomains }),
					...(req.excludeDomains?.length && { excludeDomains: req.excludeDomains }),
					...(since && { startPublishedDate: since }),
				}),
			},
			req.signal,
		);
		return ((data.results ?? []) as any[]).slice(0, req.count).map((r) => ({
			title: r.title ?? "",
			url: r.url ?? "",
			snippet: (r.highlights ?? []).join(" … ") || (r.text ?? ""),
			date: r.publishedDate ?? undefined,
		}));
	},
};

// ── Keyless fallback ─────────────────────────────────────────────────
//
// The pi extension ends with a Python metasearch (ddgs). Here it is TypeScript and limited to an official API:
// scraping DuckDuckGo / Bing / Brave HTML from this machine gets a bot challenge (DuckDuckGo), degraded results
// that ignore site: filters (Bing) or redirect-only links (Brave). Wikipedia's search API is the keyless last resort.

async function fetchText(url: string, signal?: AbortSignal): Promise<string> {
	let res: Response;
	try {
		res = await fetch(url, {
			headers: { "User-Agent": "lion-web-search/1.0 (local tool)" },
			signal: AbortSignal.any([AbortSignal.timeout(TIMEOUT_MS), ...(signal ? [signal] : [])]),
		});
	} catch (err) {
		throw new ProviderError(err instanceof Error ? err.message : String(err));
	}
	if (!res.ok) throw new ProviderError(`HTTP ${res.status}`);
	return res.text();
}

const clean = (s: string | null | undefined) => (s ?? "").replace(/\s+/g, " ").trim();

/** Wikipedia's search API (keyless, official). Recency and excluded sites do not apply to it. */
const wikipedia: SearchProvider = {
	name: "wikipedia",
	async search(req) {
		if (req.includeDomains?.length && !req.includeDomains.some((d) => d.endsWith("wikipedia.org"))) return [];
		const url = new URL("https://en.wikipedia.org/w/api.php");
		for (const [k, v] of Object.entries({
			action: "query", list: "search", srsearch: req.query, srlimit: String(req.count), format: "json", origin: "*",
		})) url.searchParams.set(k, v);
		const data = JSON.parse(await fetchText(url.toString(), req.signal));
		return ((data.query?.search ?? []) as any[]).map((r) => ({
			title: r.title ?? "",
			url: `https://en.wikipedia.org/wiki/${encodeURIComponent(String(r.title ?? "").replace(/ /g, "_"))}`,
			snippet: clean(String(r.snippet ?? "").replace(/<[^>]+>/g, "")),
			date: r.timestamp ? String(r.timestamp).slice(0, 10) : undefined,
		}));
	},
};

// ── API keys ─────────────────────────────────────────────────────────
//
// Keys live in the system keyring (libsecret), looked up from inside the pi process so
// they are never exported to the shell environment that the agent's bash commands see.
// Store one with:  secret-tool store --label="pi web_search: EXA_API_KEY" service pi-web-search key EXA_API_KEY

const KEYRING_SERVICE = "pi-web-search";
const keyCache = new Map<string, string>();

function keyringLookup(name: string): Promise<string | undefined> {
	return new Promise((resolve) => {
		execFile(
			"secret-tool",
			["lookup", "service", KEYRING_SERVICE, "key", name],
			{ timeout: 5000 },
			// Exits non-zero when there's no such item (or no keyring/secret-tool)
			(err, stdout) => resolve(err ? undefined : stdout.trim() || undefined),
		);
	});
}

/** Keyring first, then an environment variable of the same name as a fallback. */
async function resolveKey(name: string): Promise<string | undefined> {
	const cached = keyCache.get(name);
	if (cached) return cached;
	const key = (await keyringLookup(name)) ?? process.env[name];
	// Only cache hits, so a key added mid-session is picked up on the next search
	if (key) keyCache.set(name, key);
	return key;
}

/** Default order. Override with PI_WEB_SEARCH_PROVIDERS="exa,wikipedia" (comma-separated names). */
export const PROVIDERS: SearchProvider[] = [tinyfish, langsearch, exa, wikipedia];

export interface SearchOutcome {
	provider: string;
	results: SearchResult[];
	/** Providers that were skipped or failed before the one that answered. */
	skipped: string[];
}

/** Try providers in order; skip ones without a key, fall through on errors and empty results. */
export async function searchWithFallback(req: SearchRequest): Promise<SearchOutcome> {
	const order = process.env.PI_WEB_SEARCH_PROVIDERS?.split(",").map((s) => s.trim()).filter(Boolean);
	const providers = order ? PROVIDERS.filter((p) => order.includes(p.name)) : PROVIDERS;
	if (order) providers.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));

	const skipped: string[] = [];
	for (const provider of providers) {
		if (req.signal?.aborted) throw new Error("Aborted");
		const key = provider.keyEnv ? await resolveKey(provider.keyEnv) : undefined;
		if (provider.keyEnv && !key) {
			skipped.push(`${provider.name}: no ${provider.keyEnv} in keyring`);
			continue;
		}
		try {
			const results = await provider.search(req, key);
			if (results.length) return { provider: provider.name, results, skipped };
			skipped.push(`${provider.name}: no results`);
		} catch (err) {
			skipped.push(`${provider.name}: ${err instanceof Error ? err.message : String(err)}`);
		}
	}
	throw new Error(`All search providers failed:\n  ${skipped.join("\n  ")}`);
}
