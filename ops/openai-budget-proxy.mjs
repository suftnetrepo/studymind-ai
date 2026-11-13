// COPY of studymind-web/tests/fixtures/openai-budget-proxy.mjs (source of truth: the Web repo, where it has unit tests).
// Keep the two identical; the staging compose file runs this copy.
// Budget-enforcing forwarding proxy for CONTROLLED live-OpenAI tests of the StudyMind AI backend.
//
// Why: the backend records no per-call token usage on the canonical path, retries internally, and an OpenAI-side
// cap may not be configured. Every provider call the backend makes passes through here, so this is where a ceiling
// can be enforced reliably (including retries and embeddings) and where a usage ledger can be produced.
//
//   backend  --(OPENAI_API_BASE/OPENAI_BASE_URL=http://127.0.0.1:18300/v1, dummy key)-->  this proxy  -->  OpenAI
//
// The REAL key is read from PROXY_OPENAI_API_KEY (environment only) and is injected here. It is never written to a
// file, logged, returned by /__ledger, or handed to the backend process.
//
// Limits (all refused locally with HTTP 429 BEFORE anything is forwarded; refusals cost nothing):
//   MAX_CHAT_CALLS (30)   chat.completions calls actually forwarded     MAX_EMBED_CALLS (40)  embedding calls
//   MAX_USD (5.00)        worst-case reservation + measured spend must stay under this
//   CONCURRENCY is fixed at 1: requests are queued and forwarded one at a time. Upstream timeout: UPSTREAM_TIMEOUT_MS (60000).
//   The proxy itself never retries.
//
//   LEDGER_FILE=<path> persists the ledger (atomic write after every call/refusal) and reloads it at start, so the ceiling and
//   call caps survive restarts. HOST (default 127.0.0.1) is the bind address; /__ledger has no auth, keep it on a private network.
//   GET /__ledger   totals + one row per forwarded call (no headers, no bodies)      DELETE /__ledger  reset
import http from 'node:http';
import https from 'node:https';
import { existsSync, readFileSync, renameSync, writeFileSync } from 'node:fs';

const num = (k, d) => (process.env[k] !== undefined && process.env[k] !== '' ? Number(process.env[k]) : d);
export const config = () => ({
  port: num('PORT', 18300),
  host: process.env.HOST || '127.0.0.1',            // 127.0.0.1 for a local run; 0.0.0.0 only inside a private container network
  ledgerFile: process.env.LEDGER_FILE || '',        // persist the ledger so a restart cannot reset the ceiling
  upstream: new URL(process.env.UPSTREAM_URL || 'https://api.openai.com'),
  key: process.env.PROXY_OPENAI_API_KEY || '',
  maxChat: num('MAX_CHAT_CALLS', 30),
  maxEmbed: num('MAX_EMBED_CALLS', 40),
  maxUsd: num('MAX_USD', 5.0),
  timeoutMs: num('UPSTREAM_TIMEOUT_MS', 60_000),
  // USD per 1M tokens [input, output]. Chat prices match the backend table; the embedding price is from a
  // third-party source (OpenAI's pricing page was not confirmed) - treat estimates as estimates.
  prices: { 'gpt-4.1-mini': [0.4, 1.6], 'gpt-4.1': [2, 8], 'gpt-4o-mini': [0.15, 0.6], 'gpt-4o': [2.5, 10], 'text-embedding-3-large': [0.13, 0], 'text-embedding-3-small': [0.02, 0] },
});

export function priceOf(prices, model) {
  if (!model) return [Math.max(...Object.values(prices).map((p) => p[0])), Math.max(...Object.values(prices).map((p) => p[1]))]; // unknown => priciest
  if (prices[model]) return prices[model];
  const hit = Object.keys(prices).filter((n) => model.startsWith(`${n}-`)).sort((a, b) => b.length - a.length)[0]; // dated snapshots
  return hit ? prices[hit] : priceOf(prices, undefined);
}

export function createProxy(cfg = config()) {
  const ledger = { calls: [], refused: 0, spentUsd: 0 };
  if (cfg.ledgerFile && existsSync(cfg.ledgerFile)) {          // reload spend after a restart; a corrupt file fails closed
    try {
      const saved = JSON.parse(readFileSync(cfg.ledgerFile, 'utf8'));
      if (!Array.isArray(saved.calls) || !Number.isFinite(saved.spentUsd) || !Number.isFinite(saved.refused)) throw new Error('bad shape');
      Object.assign(ledger, { calls: saved.calls, refused: saved.refused, spentUsd: saved.spentUsd });
    } catch { throw new Error(`budget proxy: ledger file ${cfg.ledgerFile} is unreadable; refusing to start with an unknown spend`); }
  }
  const persist = () => { if (!cfg.ledgerFile) return; const tmp = `${cfg.ledgerFile}.tmp`; writeFileSync(tmp, JSON.stringify(ledger), { mode: 0o600 }); renameSync(tmp, cfg.ledgerFile); };
  let queue = Promise.resolve();
  const counts = () => ({ chat: ledger.calls.filter((c) => c.kind === 'chat').length, embed: ledger.calls.filter((c) => c.kind === 'embedding').length });
  const reply = (res, status, body) => { res.writeHead(status, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
  const refuse = (res, why) => { ledger.refused++; persist(); reply(res, 429, { error: { message: `budget proxy refused: ${why}`, type: 'budget_proxy', code: 'budget_proxy_limit' } }); };

  const forward = (path, method, headers, raw, res, kind, model, stream, reservedUsd) => new Promise((resolve) => {
    const u = cfg.upstream; const lib = u.protocol === 'https:' ? https : http;
    const row = { at: new Date().toISOString(), kind, model, stream, inputTokens: 0, outputTokens: 0, usd: 0, status: 0, ms: 0, reservedUsd };
    const t0 = Date.now(); let settled = false;
    const done = () => { if (settled) return; settled = true; row.ms = Date.now() - t0; ledger.calls.push(row); persist(); resolve(); };
    const up = lib.request({ hostname: u.hostname, port: u.port || (u.protocol === 'https:' ? 443 : 80), path, method,
      headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(raw), ...(cfg.key ? { authorization: `Bearer ${cfg.key}` } : {}) }, timeout: cfg.timeoutMs }, (r) => {
      row.status = r.statusCode; const chunks = [];
      res.writeHead(r.statusCode, { 'Content-Type': r.headers['content-type'] || 'application/json' });
      r.on('data', (c) => { chunks.push(c); res.write(c); });
      r.on('end', () => {
        res.end();
        const text = Buffer.concat(chunks).toString();
        let usage; model = row.model;
        try {
          if (stream) { for (const line of text.split('\n')) if (line.startsWith('data: {')) { const j = JSON.parse(line.slice(6)); if (j.usage) usage = j.usage; if (j.model) row.model = j.model; } }
          else { const j = JSON.parse(text); usage = j.usage; if (j.model) row.model = j.model; }
        } catch { /* unparseable: cost stays reserved-as-unknown below */ }
        const [pi, po] = priceOf(cfg.prices, row.model);
        if (usage) { row.inputTokens = usage.prompt_tokens || 0; row.outputTokens = usage.completion_tokens || 0; row.usd = (row.inputTokens * pi + row.outputTokens * po) / 1e6; row.measured = true; }
        else if (r.statusCode < 300) { row.measured = false; row.usd = row.reservedUsd || 0; } // success without usage: charge the reservation
        ledger.spentUsd += row.usd; done();
      });
    });
    up.on('timeout', () => up.destroy(new Error('upstream timeout')));
    up.on('error', () => { row.status = row.status || 502; if (!res.headersSent) reply(res, 502, { error: { message: 'upstream error', type: 'budget_proxy' } }); else res.end(); row.usd = row.reservedUsd || 0; ledger.spentUsd += row.usd; done(); });
    up.end(raw);
  });

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/__ledger') {
      if (req.method === 'DELETE') { ledger.calls.length = 0; ledger.refused = 0; ledger.spentUsd = 0; persist(); return reply(res, 200, { cleared: true }); }
      const c = counts(); const byModel = {};
      for (const k of ledger.calls) { const m = (byModel[k.model] ||= { calls: 0, inputTokens: 0, outputTokens: 0, usd: 0 }); m.calls++; m.inputTokens += k.inputTokens; m.outputTokens += k.outputTokens; m.usd += k.usd; }
      return reply(res, 200, { limits: { maxChat: cfg.maxChat, maxEmbed: cfg.maxEmbed, maxUsd: cfg.maxUsd, concurrency: 1 }, counts: c, refused: ledger.refused,
        spentUsd: ledger.spentUsd, inputTokens: ledger.calls.reduce((a, k) => a + k.inputTokens, 0), outputTokens: ledger.calls.reduce((a, k) => a + k.outputTokens, 0), byModel, calls: ledger.calls });
    }
    const p = []; req.on('data', (c) => p.push(c));
    req.on('end', () => {
      if (!cfg.key && cfg.upstream.hostname === 'api.openai.com') return refuse(res, 'PROXY_OPENAI_API_KEY is not set');
      const raw = Buffer.concat(p).toString(); let body = {}; try { body = JSON.parse(raw || '{}'); } catch { return reply(res, 400, { error: { message: 'invalid JSON' } }); }
      const isChat = url.pathname.endsWith('/chat/completions'); const isEmbed = url.pathname.endsWith('/embeddings');
      if (!isChat && !isEmbed) return reply(res, 404, { error: { message: 'budget proxy only forwards chat.completions and embeddings' } });
      // Everything below runs one request at a time, so counts and reservations are exact.
      queue = queue.then(async () => {
        const c = counts(); const kind = isChat ? 'chat' : 'embedding';
        if (isChat && c.chat >= cfg.maxChat) return refuse(res, `chat call limit ${cfg.maxChat} reached`);
        if (isEmbed && c.embed >= cfg.maxEmbed) return refuse(res, `embedding call limit ${cfg.maxEmbed} reached`);
        const [pi, po] = priceOf(cfg.prices, body.model);
        // Worst case: 1 token per character of input (conservative, same assumption as the backend) + the output cap.
        const inputs = isEmbed ? JSON.stringify(body.input ?? '') : JSON.stringify(body.messages ?? '');
        const outCap = isChat ? Math.min(body.max_tokens ?? body.max_completion_tokens ?? 4096, 4096) : 0;
        const reservedUsd = (inputs.length * pi + outCap * po) / 1e6;
        if (ledger.spentUsd + reservedUsd > cfg.maxUsd) return refuse(res, `would exceed the $${cfg.maxUsd} ceiling (spent $${ledger.spentUsd.toFixed(4)}, this call could cost $${reservedUsd.toFixed(4)})`);
        let fwd = raw;
        if (isChat && body.stream) fwd = JSON.stringify({ ...body, stream_options: { ...(body.stream_options || {}), include_usage: true } }); // so streamed calls report usage
        if (isChat && body.max_tokens === undefined && body.max_completion_tokens === undefined) fwd = JSON.stringify({ ...JSON.parse(fwd), max_tokens: outCap });
        await forward(url.pathname + url.search, 'POST', req.headers, fwd, res, kind, body.model, !!body.stream, reservedUsd);
      }).catch(() => { try { res.end(); } catch { /* ignore */ } });
    });
  });
  return { server, ledger };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const cfg = config(); const { server } = createProxy(cfg);
  server.listen(cfg.port, cfg.host, () => console.log(`openai-budget-proxy on ${cfg.host}:${cfg.port} -> ${cfg.upstream.origin} | chat<=${cfg.maxChat} embed<=${cfg.maxEmbed} usd<=${cfg.maxUsd} concurrency=1 | ledger ${cfg.ledgerFile || 'in memory (resets on restart)'} | key ${cfg.key ? 'set (hidden)' : 'NOT set'}`));
}
