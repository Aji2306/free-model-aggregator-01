// Free Model Aggregator server: intelligent quality routing, failover, concurrency serialization, and multi-wire support.

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
  ProviderLocker,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsFromGemini,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIQueryToAnthropic,
  openAIToAnthropic,
  openAIEventsFromGemini,
  requestToGemini,
  geminiToOpenAI,
  geminiToAnthropic,
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
const providerLocker = new ProviderLocker();

let idCounter = 1;
const nextChatId = () => `chatcmpl-${(idCounter++).toString(36)}`;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Find all free candidates meeting requirements, sorted by rating_milli descending (highest quality first). */
function candidateRoutesFor(requires: Requirement): Candidate[] {
  const candidates: Candidate[] = [];
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
  // Sort by rating descending, then provider id, then model id for deterministic tie-breaking
  candidates.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return candidates;
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
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
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

/** Call a provider upstream with concurrency serialization and failover support. */
async function callUpstream(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json
): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };

  let url = '';
  let bodyJson: Json = {};

  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      bodyJson = { ...clientBody, model: candidate.model.id };
    } else {
      bodyJson = anthropicToOpenAI(clientBody, candidate.model.id);
    }
    if (bodyJson.stream === true) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') {
      bodyJson = { ...clientBody, model: candidate.model.id };
    } else {
      bodyJson = openAIQueryToAnthropic(clientBody, candidate.model.id);
    }
    if (bodyJson.stream === true) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'gemini') {
    const isStream = clientBody.stream === true;
    const action = isStream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${candidate.model.id}:${action}`;
    if (provider.auth === 'x-goog-api-key' && provider.key !== '') {
      url += (url.includes('?') ? '&' : '?') + `key=${provider.key}`;
    }
    bodyJson = requestToGemini(clientBody, clientApi);
    if (isStream) headers.accept = 'text/event-stream';
  }

  return providerLocker.run(provider.id, async () => {
    return fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(bodyJson),
    });
  });
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

async function handleRequest(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }

  const requires = requirementsFromBody(body);
  const candidates = candidateRoutesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }

  let winningCandidate: Candidate | null = null;
  let upstream: Response | null = null;
  let lastStatus = 502;
  let lastText = '';

  for (const candidate of candidates) {
    try {
      const resUp = await callUpstream(candidate, api, body);
      if (resUp.status === 402 || resUp.status === 429 || resUp.status >= 500) {
        lastStatus = resUp.status;
        lastText = await resUp.text().catch(() => '');
        continue;
      }
      winningCandidate = candidate;
      upstream = resUp;
      break;
    } catch (err) {
      lastText = (err as Error).message;
      continue;
    }
  }

  if (winningCandidate === null || upstream === null || !upstream.ok) {
    const status = upstream ? upstream.status : lastStatus;
    const errText = upstream ? await upstream.text().catch(() => lastText) : lastText;
    if (api === 'anthropic') {
      sendJson(res, status, anthropicError('api_error', `provider HTTP ${status}: ${errText.slice(0, 500)}`));
    } else {
      sendJson(res, status, openAIError(`provider HTTP ${status}: ${errText.slice(0, 500)}`, 'api_error', null));
    }
    return;
  }

  const modelId = winningCandidate.model.id;
  const upstreamWire = winningCandidate.probe.provider.wire;
  const isStream = body.stream === true;

  if (api === 'openai') {
    if (isStream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (upstreamWire === 'gemini') {
        for await (const ev of openAIEventsFromGemini(upstream.body, modelId)) res.write(ev);
        res.end();
      } else if (upstreamWire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else {
        await pipeStream(upstream.body, res);
      }
      return;
    }

    const answer: unknown = await upstream.json();
    const answerRec = isRecord(answer) ? answer : {};
    if (upstreamWire === 'gemini') {
      sendJson(res, 200, geminiToOpenAI(answerRec, modelId));
    } else if (upstreamWire === 'anthropic') {
      const text = textOf(answerRec.content);
      const usage = isRecord(answerRec.usage) ? answerRec.usage : {};
      sendJson(res, 200, {
        id: nextChatId(),
        object: 'chat.completion',
        created: FIXED_OPENAI_CREATED_CONST,
        model: modelId,
        choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }],
        usage: {
          prompt_tokens: numberOrZero(usage.input_tokens),
          completion_tokens: numberOrZero(usage.output_tokens),
          total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
        },
      });
    } else {
      sendJson(res, 200, answerRec);
    }
  } else {
    if (isStream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (upstreamWire === 'openai') {
        for await (const ev of anthropicEventsFromOpenAI(upstream.body, modelId)) res.write(ev);
        res.end();
      } else if (upstreamWire === 'gemini') {
        for await (const ev of anthropicEventsFromGemini(upstream.body, modelId)) res.write(ev);
        res.end();
      } else {
        await pipeStream(upstream.body, res);
      }
      return;
    }

    const answer: unknown = await upstream.json();
    const answerRec = isRecord(answer) ? answer : {};
    if (upstreamWire === 'openai') {
      sendJson(res, 200, openAIToAnthropic(answerRec, modelId));
    } else if (upstreamWire === 'gemini') {
      sendJson(res, 200, geminiToAnthropic(answerRec, modelId));
    } else {
      sendJson(res, 200, answerRec);
    }
  }
}

const FIXED_OPENAI_CREATED_CONST = 1767225600;

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleRequest(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleRequest(req, res, 'anthropic');
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
        // retry
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

void start().catch((err: unknown) => {
  process.stderr.write(`aggregator: fatal: ${(err as Error).stack ?? String(err)}\n`);
  process.exit(1);
});
