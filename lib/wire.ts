// Client API translation and wire handling for OpenAI, Anthropic, and Gemini upstream wires.

import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
let deterministicIdCounter = 1;
const nextIdStr = () => (deterministicIdCounter++).toString(36);

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // ignore
    }
  }
  return {};
}

export function stopReasonOf(finish: unknown, upstreamWire: string = 'openai'): string {
  if (upstreamWire === 'gemini') {
    if (finish === 'MAX_TOKENS') return 'max_tokens';
    return 'end_turn';
  }
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Convert Anthropic request body to OpenAI request body */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Convert OpenAI request body to Anthropic request body */
export function openAIQueryToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = stringOf(raw.role);
    const content = raw.content;
    if (role === 'system') {
      system = textOf(content);
      continue;
    }
    if (role === 'user') {
      if (typeof content === 'string') {
        messages.push({ role: 'user', content });
      } else if (Array.isArray(content)) {
        const parts: Json[] = [];
        for (const p of content) {
          if (!isRecord(p)) continue;
          if (p.type === 'text' && typeof p.text === 'string') {
            parts.push({ type: 'text', text: p.text });
          } else if (p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
            const url = p.image_url.url;
            if (url.startsWith('data:')) {
              const match = /^data:([^;]+);base64,(.+)$/.exec(url);
              if (match) {
                parts.push({
                  type: 'image',
                  source: { type: 'base64', media_type: match[1], data: match[2] },
                });
              }
            } else {
              parts.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
        messages.push({ role: 'user', content: parts });
      }
      continue;
    }
    if (role === 'assistant') {
      const contentBlocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') contentBlocks.push({ type: 'text', text });
      const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of toolCalls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        contentBlocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: contentBlocks.length > 0 ? contentBlocks : text });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: textOf(content),
          },
        ],
      });
      continue;
    }
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  else out.max_tokens = 4096;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.map((t) => {
      if (isRecord(t) && isRecord(t.function)) {
        const fn = t.function;
        return {
          name: stringOf(fn.name),
          description: stringOf(fn.description),
          input_schema: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
        };
      }
      return t;
    });
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Convert client request (OpenAI or Anthropic) to Gemini request body */
export function requestToGemini(body: Json, api: 'openai' | 'anthropic'): Json {
  const anthropicBody = api === 'openai' ? openAIQueryToAnthropic(body, 'gemini') : body;
  const contents: Json[] = [];
  let systemInstruction: Json | undefined = undefined;

  if (typeof anthropicBody.system === 'string' && anthropicBody.system !== '') {
    systemInstruction = { parts: [{ text: anthropicBody.system }] };
  }

  const messages = Array.isArray(anthropicBody.messages) ? anthropicBody.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = m.content;
    if (typeof content === 'string') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/jpeg', data: src.data } });
          } else if (src.type === 'url' && typeof src.url === 'string') {
            parts.push({ fileData: { fileUri: src.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: 'tool', response: { output: block.content } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }

  const generationConfig: Json = {};
  if (typeof anthropicBody.max_tokens === 'number') generationConfig.maxOutputTokens = anthropicBody.max_tokens;
  if (Array.isArray(anthropicBody.stop_sequences)) generationConfig.stopSequences = anthropicBody.stop_sequences;
  const isJson = isRecord(anthropicBody.response_format) && anthropicBody.response_format.type === 'json_object';
  if (isJson) generationConfig.responseMimeType = 'application/json';

  const out: Json = { contents };
  if (systemInstruction) out.systemInstruction = systemInstruction;
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;

  if (Array.isArray(anthropicBody.tools)) {
    const functionDeclarations = anthropicBody.tools.map((t: any) => ({
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    }));
    out.tools = [{ functionDeclarations }];
  }

  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall)) {
      const fc = p.functionCall;
      toolCalls.push({
        id: `call_${nextIdStr()}`,
        type: 'function',
        function: {
          name: stringOf(fc.name),
          arguments: JSON.stringify(isRecord(fc.args) ? fc.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: `chatcmpl-${nextIdStr()}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: text !== '' ? text : null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: cand.finishReason === 'MAX_TOKENS' ? 'length' : toolCalls.length > 0 ? 'tool_calls' : 'stop',
      },
    ],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

export function geminiToAnthropic(body: Json, model: string): Json {
  const openAIResp = geminiToOpenAI(body, model);
  return openAIToAnthropic(openAIResp, model);
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };

  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            }),
          );
        }
        out.push(
          sseEvent('content_block_delta', {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: delta.content },
          }),
        );
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(
            sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index,
              delta: { type: 'input_json_delta', partial_json: fn.arguments },
            }),
          );
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let textIndex = -1;
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    ];
  };

  textIndex = 0;

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usageMetadata)) {
        inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand === null) continue;
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      out.push(...startEvents());
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          out.push(
            sseEvent('content_block_delta', {
              type: 'content_block_delta',
              index: textIndex,
              delta: { type: 'text_delta', text: p.text },
            }),
          );
        }
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: 'end_turn', stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const chunkOf = (content: string | null, finish: string | null) => ({
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: content !== null ? { content } : {}, finish_reason: finish }],
  });

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand === null) continue;
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          out.push(`data: ${JSON.stringify(chunkOf(p.text, null))}\n\n`);
        }
      }
      if (cand.finishReason) {
        out.push(`data: ${JSON.stringify(chunkOf(null, cand.finishReason === 'MAX_TOKENS' ? 'length' : 'stop'))}\n\n`);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const ev of eventsOfFrame(frame)) yield ev;
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';
}
