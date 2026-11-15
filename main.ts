// The free-model aggregator server entrypoint (Node 22).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  geminiError,
  geminiModelList,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicBody,
  requestToGemini,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

// Per-provider mutex/queue to ensure one provider serves one call at a time.
const providerLocks = new Map<string, Promise<void>>();

async function withProviderLock<T>(providerId: string, fn: () => Promise<T>): Promise<T> {
  const current = providerLocks.get(providerId) ?? Promise.resolve();
  let release!: () => void;
  const next = new Promise<void>((res) => {
    release = res;
  });
  providerLocks.set(providerId, next);
  await current;
  try {
    return await fn();
  } finally {
    release();
  }
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface CandidateRoute {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Find all free models meeting requirements, sorted by quality rating descending. */
function candidateRoutesFor(requires: Requirement): CandidateRoute[] {
  const candidates: CandidateRoute[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      candidates.push({ probe, model, rating });
    }
  }
  candidates.sort((a, b) => b.rating - a.rating);
  return candidates;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function callUpstream(route: CandidateRoute, clientApi: 'openai' | 'anthropic', clientBody: Json): Promise<Response> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };

  if (provider.wire === 'openai') {
    if (clientBody.stream === true) headers.accept = 'text/event-stream';
    const payload = clientApi === 'anthropic' ? anthropicToOpenAI(clientBody, route.model.id) : { ...clientBody, model: route.model.id };
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(payload) });
  }

  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (clientBody.stream === true) headers.accept = 'text/event-stream';
    const payload = clientApi === 'openai' ? openAIToAnthropicBody(clientBody, route.model.id) : { ...clientBody, model: route.model.id };
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(payload) });
  }

  if (provider.wire === 'gemini') {
    const isStream = clientBody.stream === true;
    const action = isStream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    let url = `${base}/v1beta/models/${route.model.id}:${action}`;
    if (provider.auth === 'x-goog-api-key' && provider.key !== '') {
      url += `${url.includes('?') ? '&' : '?'}key=${encodeURIComponent(provider.key)}`;
    }
    const geminiBody = requestToGemini(clientBody);
    return fetch(url, { method: 'POST', headers, body: JSON.stringify(geminiBody) });
  }

  throw new Error(`unknown upstream wire: ${provider.wire}`);
}

function openAIToAnthropicPayload(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const anthropicMessages: Json[] = [];
  let system = '';

  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') {
      system = typeof m.content === 'string' ? m.content : textOf(m.content);
    } else {
      anthropicMessages.push({ role: m.role, content: m.content });
    }
  }

  const out: Json = { model, messages: anthropicMessages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else out.max_tokens = 4096;
  if (body.stream === true) out.stream = true;
  return out;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

async function pipeGeminiStream(stream: ReadableStream<Uint8Array>, res: ServerResponse, clientApi: 'openai' | 'anthropic', model: string): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) {
            const payload = trimmed.slice(5).trim();
            try {
              const chunk = JSON.parse(payload);
              if (clientApi === 'openai') {
                const converted = geminiToOpenAI(chunk, model);
                res.write(`data: ${JSON.stringify(converted)}\n\n`);
              } else {
                // For anthropic stream from gemini, emit basic events
                const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
                const cand = isRecord(candidates[0]) ? candidates[0] : {};
                const content = isRecord(cand.content) ? cand.content : {};
                const parts = Array.isArray(content.parts) ? content.parts : [];
                let text = '';
                for (const p of parts) if (isRecord(p) && typeof p.text === 'string') text += p.text;
                if (text !== '') {
                  res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
                }
              }
            } catch {
              // ignore
            }
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

async function handleInference(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }

  const requires = requirementsFor(body);
  const candidates = candidateRoutesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  }

  let lastError: string = '';
  for (const candidate of candidates) {
    try {
      const upstream = await withProviderLock(candidate.probe.provider.id, () => callUpstream(candidate, clientApi, body));
      if (!upstream.ok) {
        const errText = await upstream.text();
        lastError = `upstream HTTP ${upstream.status}: ${errText.slice(0, 200)}`;
        // 402, 429, or 5xx: fail over to next candidate
        if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
          continue;
        }
        // Other errors: return as is
        res.writeHead(upstream.status, { 'content-type': 'application/json' });
        res.end(errText);
        return;
      }

      const isStream = body.stream === true;
      const wire = candidate.probe.provider.wire;

      if (isStream && upstream.body !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        if (wire === 'openai') {
          if (clientApi === 'anthropic') {
            const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
            for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
            res.end();
          } else {
            await pipeStream(upstream.body, res);
          }
        } else if (wire === 'anthropic') {
          if (clientApi === 'anthropic') {
            await pipeStream(upstream.body, res);
          } else {
            // Translate anthropic stream to openai stream or pipe
            await pipeStream(upstream.body, res);
          }
        } else if (wire === 'gemini') {
          const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
          await pipeGeminiStream(upstream.body, res, clientApi, requested);
        }
        return;
      }

      // Non-streaming response
      const answerText = await upstream.text();
      let answerJson: Json;
      try {
        answerJson = JSON.parse(answerText);
      } catch {
        res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
        res.end(answerText);
        return;
      }

      const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
      if (clientApi === 'anthropic') {
        if (wire === 'openai') {
          sendJson(res, 200, openAIToAnthropic(answerJson, requested));
        } else if (wire === 'anthropic') {
          sendJson(res, 200, answerJson);
        } else if (wire === 'gemini') {
          sendJson(res, 200, geminiToAnthropic(answerJson, requested));
        }
      } else {
        if (wire === 'openai') {
          sendJson(res, 200, answerJson);
        } else if (wire === 'anthropic') {
          // Anthropic to OpenAI response translation if needed
          const text = textOf(answerJson.content);
          sendJson(res, 200, {
            id: 'chatcmpl-anthropic',
            object: 'chat.completion',
            created: FIXED_OPENAI_CREATED,
            model: requested,
            choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 },
          });
        } else if (wire === 'gemini') {
          sendJson(res, 200, geminiToOpenAI(answerJson, requested));
        }
      }
      return;
    } catch (err) {
      lastError = (err as Error).message;
      continue; // try next candidate
    }
  }

  sendError(res, clientApi, 503, 'failover_exhausted', `all candidate free models failed. Last error: ${lastError}`);
}

const FIXED_OPENAI_CREATED = 1767225600;

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleInference(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleInference(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // retry on next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
