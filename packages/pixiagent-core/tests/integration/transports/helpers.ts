/**
 * Shared helpers for transport integration tests.
 *
 * The test conversation covers:
 *   Turn 1 – simple chat (system prompt + greeting)
 *   Turn 2 – tool call with no args  (future_weather)
 *   Turn 3 – tool call with args     (stock_ohlc)
 *   Turn 4 – reasoning               (math word problem, thinkEffort='medium')
 */

import { expect } from 'vitest';
import {
  ApiModes,
  SessionMessage,
  ThinkingPart,
  ToolCallPart,
  ToolResultPart,
  TextPart,
  RawMessageType,
  UsageStats,
} from '../../../src/message';
import { ProviderTransport, ModelOptions, StreamCallbacks } from '../../../src/transports/base';
import { fakeToolset, executeToolCall } from '../poc/tools';

// ── option builders ──────────────────────────────────────────────────────────

export function baseOptions(model: string, extra?: Partial<ModelOptions>): ModelOptions {
  return {
    model,
    systemPrompt: 'You are a helpful assistant. Answer concisely and accurately.',
    maxTokens: 2048,
    tools: fakeToolset.tools.map((t) => t.definition),
    ...extra,
  };
}

// ── callback collector ───────────────────────────────────────────────────────

export type CollectedCallbacks = {
  textChunks: string[];
  textChunkFlags: Array<'begin' | 'end' | undefined>;
  thinkingChunks: string[];
  thinkingChunkFlags: Array<'begin' | 'end' | undefined>;
  thinkingFull: string[];
  toolUseEvents: Array<{ name: string; delta: string }>;
  errors: Error[];
};

export function makeCallbacks(): { callbacks: StreamCallbacks; collected: CollectedCallbacks } {
  const collected: CollectedCallbacks = {
    textChunks: [],
    textChunkFlags: [],
    thinkingChunks: [],
    thinkingChunkFlags: [],
    thinkingFull: [],
    toolUseEvents: [],
    errors: [],
  };

  const callbacks: StreamCallbacks = {
    onTextChunk: (delta, flag) => {
      collected.textChunks.push(delta);
      collected.textChunkFlags.push(flag);
    },
    onThinkingChunk: (delta, flag) => {
      collected.thinkingChunks.push(delta);
      collected.thinkingChunkFlags.push(flag);
    },
    onThinking: (text) => {
      collected.thinkingFull.push(text);
    },
    onToolUse: (name, delta) => {
      collected.toolUseEvents.push({ name, delta });
    },
    onError: (err) => {
      collected.errors.push(err);
    },
  };

  return { callbacks, collected };
}

// ── message history helpers ───────────────────────────────────────────────────

export function sessionMessagesToRawMessages(
  transport: ProviderTransport<RawMessageType>,
  messages: SessionMessage[],
): RawMessageType[] {
  return messages.flatMap((msg) => {
    const raw = transport.convertToRawMessage(msg);
    return Array.isArray(raw) ? raw : [raw];
  });
}

/**
 * Execute all tool calls present in an InternalMessage and return tool SessionMessages
 * ready to be appended to the conversation history.
 */
export async function handleToolCalls(
  transport: ProviderTransport<RawMessageType>,
  result: RawMessageType,
): Promise<SessionMessage[]> {
  const sessionMsg = transport.convertFromRawMessage(result);
  const content = sessionMsg.content;
  if (!Array.isArray(content)) return [];

  const toolCalls = content.filter((p) => p.type === 'tool_call') as ToolCallPart[];
  if (toolCalls.length === 0) return [];

  const toolResults: ToolResultPart[] = await Promise.all(
    toolCalls.map(async (tc) => {
      const output = await executeToolCall(
        tc.name,
        JSON.parse(tc.arguments || '{}'),
        fakeToolset.tools,
      );
      return {
        type: 'tool_result' as const,
        id: tc.id,
        name: tc.name,
        result: JSON.stringify(output),
      };
    }),
  );

  return [
    {
      type: 'session_message',
      role: 'tool',
      content: toolResults,
    },
  ];
}

// ── assertion helpers ─────────────────────────────────────────────────────────

export function assertUsage(result: UsageStats | undefined) {
  expect(result?.outputTokens, 'outputTokens should be > 0').toBeGreaterThan(0);
  expect(result?.inputTokens, 'inputTokens should be > 0').toBeGreaterThan(0);
}

export function assertTextStreamed(collected: CollectedCallbacks) {
  expect(
    collected.textChunks.some((c) => c.length > 0),
    'at least one non-empty text chunk expected',
  ).toBe(true);
  expect(collected.textChunkFlags, 'text chunks should start with begin').toContain('begin');
  expect(collected.textChunkFlags, 'text chunks should end with end').toContain('end');
}

export function assertTextDeltaMatchesFinalMessage(
  collected: CollectedCallbacks,
  transport: ProviderTransport<RawMessageType>,
  result: RawMessageType,
) {
  const streamedText = collected.textChunks.join('');
  const sessionMsg = transport.convertFromRawMessage(result);
  const content = sessionMsg.content;

  let finalText = '';
  if (typeof content === 'string') {
    finalText = content;
  } else if (Array.isArray(content)) {
    finalText = content
      .filter((p): p is TextPart => p.type === 'text')
      .map((p) => p.text)
      .join('');
  }

  expect(
    streamedText,
    'concatenated text deltas should equal final assistant text in message',
  ).toBe(finalText);
}

export function assertToolCallInResult(result: RawMessageType, transport: ProviderTransport<RawMessageType>, expectedToolName: string) {
  const sessionMsg = transport.convertFromRawMessage(result);
  const content = sessionMsg.content;
  expect(Array.isArray(content), 'assistant message content should be an array').toBe(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const toolCalls = (content as any[]).filter((p: any) => p.type === 'tool_call') as ToolCallPart[];
  expect(
    toolCalls.some((tc) => tc.name === expectedToolName),
    `expected tool call '${expectedToolName}' in assistant response`,
  ).toBe(true);
  return toolCalls.find((tc) => tc.name === expectedToolName)!;
}

export function assertThinkingStreamed(collected: CollectedCallbacks) {
  expect(
    collected.thinkingChunks.some((c) => c.length > 0),
    'at least one non-empty thinking chunk expected',
  ).toBe(true);
  expect(collected.thinkingChunkFlags, 'thinking chunks should start with begin').toContain('begin');
  expect(collected.thinkingChunkFlags, 'thinking chunks should end with end').toContain('end');
  expect(collected.thinkingFull.length, 'onThinking should have been called').toBeGreaterThan(0);
  expect(
    collected.thinkingFull[0].length,
    'onThinking text should be non-empty',
  ).toBeGreaterThan(0);
}

export function assertThinkingInConvertedMessage(
  transport: ProviderTransport<RawMessageType>,
  result: RawMessageType,
) {
  const sessionMsg = transport.convertFromRawMessage(result);
  const content = sessionMsg.content;
  expect(Array.isArray(content), 'assistant message content should be an array').toBe(true);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const thinkingParts = (content as any[]).filter(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (p: any) => p.type === 'thinking',
  ) as ThinkingPart[];
  expect(
    thinkingParts.length,
    'at least one ThinkingPart expected in converted message',
  ).toBeGreaterThan(0);
  expect(
    thinkingParts[0].content.length,
    'ThinkingPart content should be non-empty',
  ).toBeGreaterThan(0);
}

export function assertThinkingDeltaMatchesFinalMessage(
  collected: CollectedCallbacks,
  transport: ProviderTransport<RawMessageType>,
  result: RawMessageType,
) {
  const streamedThinking = collected.thinkingChunks.join('');
  const sessionMsg = transport.convertFromRawMessage(result);
  const content = sessionMsg.content;
  expect(Array.isArray(content), 'assistant message content should be an array').toBe(true);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const finalThinking = (content as any[])
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    .filter((p: any): p is ThinkingPart => p.type === 'thinking')
    .map((p: ThinkingPart) => p.content)
    .join('');

  expect(
    streamedThinking,
    'concatenated thinking deltas should equal final thinking content in message',
  ).toBe(finalThinking);
}

function getAllText(content?: string | Array<TextPart | ToolCallPart | ToolResultPart>): string {
  if (!content) return '';
  if (typeof content === 'string') return content;
  return content
    .filter((p): p is TextPart => p.type === 'text')
    .map((p) => p.text)
    .join('');
}

/**
 * Validate core session/raw conversion in both directions for a transport.
 */
export function assertBidirectionalConversion(
  transport: ProviderTransport<RawMessageType>,
): void {
  const userMsg: SessionMessage = {
    type: 'session_message',
    role: 'user',
    content: 'Hello transport conversion!',
  };
  const userRaw = transport.convertToRawMessage(userMsg) as RawMessageType;
  const userRoundTrip = transport.convertFromRawMessage(userRaw);
  expect(userRoundTrip.role).toBe('user');
  expect(getAllText(userRoundTrip.content as string | Array<TextPart>)).toContain('Hello');

  const assistantTextMsg: SessionMessage = {
    type: 'session_message',
    role: 'assistant',
    content: [{ type: 'text', text: 'Round-trip assistant text.' }],
  };
  const assistantRaw = transport.convertToRawMessage(assistantTextMsg) as RawMessageType;
  const assistantRoundTrip = transport.convertFromRawMessage(assistantRaw);
  expect(assistantRoundTrip.role).toBe('assistant');
  expect(getAllText(assistantRoundTrip.content as string | Array<TextPart>)).toContain(
    'assistant text',
  );

  const assistantToolCall: SessionMessage = {
    type: 'session_message',
    role: 'assistant',
    content: [
      {
        type: 'tool_call',
        id: 'call_1',
        name: 'future_weather',
        arguments: JSON.stringify({ days: 1 }),
      },
    ],
  };
  const toolCallRaw = transport.convertToRawMessage(assistantToolCall) as RawMessageType;
  const toolCallRoundTrip = transport.convertFromRawMessage(toolCallRaw);
  expect(toolCallRoundTrip.role).toBe('assistant');
  expect(Array.isArray(toolCallRoundTrip.content)).toBe(true);
  const roundTripToolCall = (toolCallRoundTrip.content as Array<ToolCallPart>).find(
    (p) => p.type === 'tool_call',
  );
  expect(roundTripToolCall?.name).toBe('future_weather');

  const toolResultMsg: SessionMessage = {
    type: 'session_message',
    role: 'tool',
    content: [
      {
        type: 'tool_result',
        id: 'call_1',
        name: 'future_weather',
        result: JSON.stringify({ temp: 23 }),
      },
    ],
  };
  const toolResultRaw = transport.convertToRawMessage(toolResultMsg);
  const toolResultRaws = Array.isArray(toolResultRaw) ? toolResultRaw : [toolResultRaw];
  const toolResultRoundTrips = toolResultRaws.map((raw) => transport.convertFromRawMessage(raw));
  const roundTripToolResults = toolResultRoundTrips.flatMap((msg) =>
    Array.isArray(msg.content)
      ? msg.content.filter((p): p is ToolResultPart => p.type === 'tool_result')
      : [],
  );
  expect(roundTripToolResults.some((p) => p.id === 'call_1')).toBe(true);

  const multiToolResultMsg: SessionMessage = {
    type: 'session_message',
    role: 'tool',
    content: [
      {
        type: 'tool_result',
        id: 'call_1',
        name: 'future_weather',
        result: JSON.stringify({ temp: 23 }),
      },
      {
        type: 'tool_result',
        id: 'call_2',
        name: 'stock_ohlc',
        result: JSON.stringify({ close: 105 }),
      },
    ],
  };
  const multiRaw = transport.convertToRawMessage(multiToolResultMsg);
  const multiRaws = Array.isArray(multiRaw) ? multiRaw : [multiRaw];
  if (transport.apiMode === ApiModes.ANTHROPIC) {
    expect(multiRaws.length).toBe(1);
  } else {
    expect(multiRaws.length).toBe(2);
  }
  const multiRoundTrips = multiRaws.map((raw) => transport.convertFromRawMessage(raw));
  const multiRoundTripResults = multiRoundTrips.flatMap((msg) =>
    Array.isArray(msg.content)
      ? msg.content.filter((p): p is ToolResultPart => p.type === 'tool_result')
      : [],
  );
  expect(multiRoundTripResults.some((p) => p.id === 'call_1')).toBe(true);
  expect(multiRoundTripResults.some((p) => p.id === 'call_2')).toBe(true);
}

// ── the 4-turn conversation runner ───────────────────────────────────────────

/**
 * Run the standard 4-turn conversation against the given transport.
 *
 * @param transport  The transport instance under test.
 * @param model      The model string to use in ModelOptions.
 * @param opts
 *   supportsReasoning – when true, Turn 4 also asserts on thinking chunks/parts.
 *   reasoningModel    – optionally a different model for Turn 4 (some dialects require it).
 */
export async function runStandardConversation(
  transport: ProviderTransport<RawMessageType>,
  model: string,
  opts: {
    supportsReasoning?: boolean;
    reasoningModel?: string;
      reasoningInFinalMessage?: boolean;
    extraOptions?: Partial<ModelOptions>;
  } = {},
) {
  const history: SessionMessage[] = [];

  // ── Turn 1: simple chat ──────────────────────────────────────────────────
  {
    const { callbacks, collected } = makeCallbacks();
    const messages: SessionMessage[] = [
      ...history,
      { type: 'session_message', role: 'user', content: 'Hello! My name is Eric. What is 2 + 2?' },
    ];

    const result = await transport.generate(
      baseOptions(model, opts.extraOptions),
      sessionMessagesToRawMessages(transport, messages),
      callbacks,
    );

    assertUsage(result.usage);
    assertTextStreamed(collected);
    assertTextDeltaMatchesFinalMessage(collected, transport, result.responseMessage);

    // push to history
    history.push(messages[messages.length - 1]);
    history.push(transport.convertFromRawMessage(result.responseMessage));
  }

  // ── Turn 2: tool call – future_weather (no args) ─────────────────────────
  {
    const { callbacks, collected } = makeCallbacks();
    const userMsg: SessionMessage = {
      type: 'session_message',
      role: 'user',
      content: 'What will the weather be like tomorrow?',
    };
    const messages: SessionMessage[] = [...history, userMsg];

    const result = await transport.generate(
      baseOptions(model, opts.extraOptions),
      sessionMessagesToRawMessages(transport, messages),
      callbacks,
    );

    assertUsage(result.usage);
    assertToolCallInResult(result.responseMessage, transport, 'future_weather');
    expect(collected.errors).toHaveLength(0);

    // execute tool and continue conversation
    const toolResults = await handleToolCalls(transport, result.responseMessage);
    history.push(userMsg);
    history.push(transport.convertFromRawMessage(result.responseMessage));
    history.push(...toolResults);

    // one more turn to get the summary
    const { callbacks: cb2, collected: col2 } = makeCallbacks();
    const followUp: SessionMessage[] = [
      ...history,
      { type: 'session_message', role: 'user', content: 'Please summarize the weather forecast.' },
    ];
    const result2 = await transport.generate(
      baseOptions(model, opts.extraOptions),
      sessionMessagesToRawMessages(transport, followUp),
      cb2,
    );
    assertUsage(result2.usage)  ;
    assertTextStreamed(col2);
    history.push(followUp[followUp.length - 1]);
    history.push(transport.convertFromRawMessage(result2.responseMessage));
  }

  // ── Turn 3: tool call – stock_ohlc (with args) ───────────────────────────
  {
    const { callbacks } = makeCallbacks();
    const userMsg: SessionMessage = {
      type: 'session_message',
      role: 'user',
      content: "What was Apple's stock price (ticker AAPL) on 2024-01-15?",
    };
    const messages: SessionMessage[] = [...history, userMsg];

    const result = await transport.generate(
      baseOptions(model, opts.extraOptions),
      sessionMessagesToRawMessages(transport, messages),
      callbacks,
    );

    assertUsage(result.usage);
    const toolCall = assertToolCallInResult(result.responseMessage, transport, 'stock_ohlc');
    const args = JSON.parse(toolCall.arguments || '{}');
    expect(args.ticker?.toUpperCase()).toBe('AAPL');
    expect(args.date).toBe('2024-01-15');

    const toolResults = await handleToolCalls(transport, result.responseMessage);
    history.push(userMsg);
    history.push(transport.convertFromRawMessage(result.responseMessage));
    history.push(...toolResults);

    const { callbacks: cb2 } = makeCallbacks();
    const followUp: SessionMessage[] = [
      ...history,
      {
        type: 'session_message',
        role: 'user',
        content: 'Thanks. Can you summarize the stock data?',
      },
    ];
    const result2 = await transport.generate(
      baseOptions(model, opts.extraOptions),
      sessionMessagesToRawMessages(transport, followUp),
      cb2,
    );
    assertUsage(result2.usage);
    history.push(followUp[followUp.length - 1]);
    history.push(transport.convertFromRawMessage(result2.responseMessage));
  }

  // ── Turn 4: reasoning ────────────────────────────────────────────────────
  if (opts.supportsReasoning) {
    const reasoningModel = opts.reasoningModel ?? model;
    const { callbacks, collected } = makeCallbacks();
    const userMsg: SessionMessage = {
      type: 'session_message',
      role: 'user',
      content:
        'Solve step by step: A train travelling at 60 mph and another at 40 mph depart from opposite ends of a 300-mile track at the same time. After how many hours do they meet, and where?',
    };

    const result = await transport.generate(
      baseOptions(reasoningModel, { ...opts.extraOptions, thinkEffort: 'medium', tools: [] }),
      sessionMessagesToRawMessages(transport, [userMsg]),
      callbacks,
    );

    assertUsage(result.usage);
    assertTextStreamed(collected);
    assertTextDeltaMatchesFinalMessage(collected, transport, result.responseMessage);
    assertThinkingStreamed(collected);
    if (opts.reasoningInFinalMessage ?? true) {
      assertThinkingInConvertedMessage(transport, result.responseMessage);
      assertThinkingDeltaMatchesFinalMessage(collected, transport, result.responseMessage);
    }
  }
}
