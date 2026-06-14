import { describe, expect, it, vi } from 'vitest';

import { AgentExecutionLoop } from './agent-loop';
import {
  ApiModes,
  ModelStopReasons,
  type RawMessageType,
  type SessionMessage,
  UsageStats,
} from './message';
import type { ModelOptions, ProviderTransport } from './transports';

function createDependencies() {
  const logger = {
    info: vi.fn(),
    warn: vi.fn(),
    debug: vi.fn(),
    error: vi.fn(),
  };

  const addMessage = vi.fn((role: string, rawMessage: unknown) => ({
    internalMessageId: `msg-${Math.random().toString(36).slice(2, 8)}`,
    previousMessageId: null,
    model: 'gpt-test',
    apiMode: ApiModes.RESPONSE,
    rawMessage,
    role,
    createdAt: '2024-01-01T00:00:00.000Z',
  }));

  const sessionThread = {
    session: { sessionId: 'session-1', parentSessionId: undefined },
    threadInfo: {
      threadId: 'thread-1',
      headMessageId: null,
      modelOptions: { model: 'gpt-test', apiMode: ApiModes.RESPONSE },
    },
    threadMessages: [],
    addMessage,
  };

  const eventEmitter = {
    executionState: {
      beforeModelRequest: vi.fn().mockResolvedValue(undefined),
      executionFinished: vi.fn().mockResolvedValue(undefined),
      iterationCompleted: vi.fn().mockResolvedValue(undefined),
      afterModelResponse: vi.fn().mockResolvedValue(undefined),
      beforeToolCallRequest: vi.fn().mockResolvedValue(undefined),
      afterToolCallResponse: vi.fn().mockResolvedValue(undefined),
      toolCallsFinished: vi.fn().mockResolvedValue(undefined),
      interrupted: vi.fn().mockResolvedValue(undefined),
    },
    messageChunk: {
      chunk: vi.fn().mockResolvedValue(undefined),
      finish: vi.fn().mockResolvedValue(undefined),
    },
  };

  return {
    logger,
    sessionThread,
    toolRegistry: { execute: vi.fn() },
    eventEmitter,
    agentOptions: {
      eventEmitter,
      maxIterations: 5,
      maxModelRequestRetries: 0,
      modelRequestTimeout: 1000,
    },
  };
}

function createLoop(
  dependencies: ReturnType<typeof createDependencies>,
  transport: ProviderTransport<RawMessageType>,
) {
  const getTransportSpy = vi
    .spyOn(
      AgentExecutionLoop as unknown as {
        getTransport: (modelOptions: ModelOptions) => ProviderTransport<RawMessageType>;
      },
      'getTransport',
    )
    .mockReturnValue(transport);

  const loop = new AgentExecutionLoop(
    dependencies.sessionThread as never,
    dependencies.toolRegistry as never,
    dependencies.agentOptions as never,
    [],
    dependencies.logger as never,
  );

  return { loop, getTransportSpy };
}

function createTransport(generateImpl: () => Promise<RawMessageType>): ProviderTransport<RawMessageType> {
  const generate = vi.fn<() => Promise<RawMessageType>>();
  generate.mockImplementation(generateImpl);

  return {
    generate,
    convertFromRawMessage: vi.fn((raw: unknown): SessionMessage => {
      const candidate = raw as { type?: string; messageId?: string; content?: unknown; modelResponseInfo?: unknown; role?: SessionMessage['role'] };
      if (candidate.type === 'session_message') {
        return candidate as SessionMessage;
      }

      return {
        messageId: candidate.messageId ?? 'msg-1',
        type: 'session_message',
        role: candidate.role ?? 'assistant',
        content: Array.isArray(candidate.content) ? candidate.content : [],
        modelResponseInfo: candidate.modelResponseInfo as SessionMessage['modelResponseInfo'],
      };
    }),
    convertToRawMessage: vi.fn((msg: SessionMessage): RawMessageType => ({
      messageId: msg.messageId,
      type: 'response_api_message',
      role: msg.role,
      content: msg.content,
    } as unknown as RawMessageType)),
  } as unknown as ProviderTransport<RawMessageType>;
}

describe('AgentExecutionLoop', () => {
  it('returns end_turn after the model stops normally', async () => {
    const dependencies = createDependencies();
    const response = {
      messageId: 'msg-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      modelResponseInfo: {
        responseId: 'resp-1',
        responseModel: 'gpt-test',
        stopReason: ModelStopReasons.STOP,
        usage: UsageStats.empty(),
      },
    };

    const transport = createTransport(async () => response as RawMessageType);
    const { loop, getTransportSpy } = createLoop(dependencies, transport);

    try {
      const modelOptions: ModelOptions = { model: 'gpt-test', apiMode: ApiModes.RESPONSE };
      const result = await loop.execute(modelOptions, { role: 'user', content: 'hi' } as never, new AbortController().signal);

      expect(result.action).toBe('execution');
      if (result.action === 'execution') {
        expect(result.stopReason).toBe('end_turn');
      }
      expect(transport.generate).toHaveBeenCalledTimes(1);
      expect(dependencies.eventEmitter.executionState.afterModelResponse).toHaveBeenCalledTimes(1);
    } finally {
      getTransportSpy.mockRestore();
    }
  });

  it('executes tool calls and continues until the final assistant response', async () => {
    const toolResult = {
      type: 'tool_result',
      id: 'call-1',
      name: 'echo',
      result: JSON.stringify({ ok: true }),
    };

    const toolResponse = {
      messageId: 'msg-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: 'call-1',
          name: 'echo',
          arguments: '{"text":"hello"}',
        },
      ],
      modelResponseInfo: {
        responseId: 'resp-tool',
        responseModel: 'gpt-test',
        stopReason: ModelStopReasons.TOOL_CALL,
        usage: UsageStats.empty(),
      },
    };

    const finalResponse = {
      messageId: 'msg-2',
      type: 'response_api_message',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      modelResponseInfo: {
        responseId: 'resp-final',
        responseModel: 'gpt-test',
        stopReason: ModelStopReasons.STOP,
        usage: UsageStats.empty(),
      },
    };

    const dependencies = createDependencies();
    const executeMock = vi.fn().mockResolvedValue(toolResult);
    dependencies.toolRegistry.execute = executeMock;

    const transport = createTransport(async () => toolResponse as unknown as RawMessageType);
    const generateMock = transport.generate as unknown as ReturnType<typeof vi.fn>;
    generateMock
      .mockResolvedValueOnce(toolResponse as unknown as RawMessageType)
      .mockResolvedValueOnce(finalResponse as unknown as RawMessageType);

    const { loop, getTransportSpy } = createLoop(dependencies, transport);

    try {
      const modelOptions: ModelOptions = {
        model: 'gpt-test',
        apiMode: ApiModes.RESPONSE,
        parallelToolCalls: false,
      };
      const result = await loop.execute(modelOptions, { role: 'user', content: 'hi' } as never, new AbortController().signal);

      expect(result.action).toBe('execution');
      if (result.action === 'execution') {
        expect(result.stopReason).toBe('end_turn');
      }
      expect(executeMock).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'echo' }),
        expect.any(Object),
      );
      expect(generateMock).toHaveBeenCalledTimes(2);
    } finally {
      getTransportSpy.mockRestore();
    }
  });

  it('stops immediately when maxIterations is reached', async () => {
    const dependencies = createDependencies();
    dependencies.agentOptions.maxIterations = 0;
    const transport = createTransport(async () => ({
      messageId: 'msg-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      modelResponseInfo: { responseId: 'resp-1', responseModel: 'gpt-test', stopReason: ModelStopReasons.STOP },
    } as unknown as RawMessageType));

    const { loop, getTransportSpy } = createLoop(dependencies, transport);

    try {
      const modelOptions: ModelOptions = { model: 'gpt-test', apiMode: ApiModes.RESPONSE };
      const result = await loop.execute(modelOptions, { role: 'user', content: 'hi' } as never, new AbortController().signal);

      expect(result.action).toBe('execution');
      if (result.action === 'execution') {
        expect(result.stopReason).toBe('max_turn_requests');
      }
      expect(transport.generate).not.toHaveBeenCalled();
    } finally {
      getTransportSpy.mockRestore();
    }
  });

  it('treats tool execution failures as tool_result errors and continues', async () => {
    const toolResult = {
      type: 'tool_result',
      id: 'call-1',
      name: 'broken',
      result: JSON.stringify({ error: 'boom' }),
      isError: true,
    };

    const toolResponse = {
      messageId: 'msg-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [{ type: 'tool_call', id: 'call-1', name: 'broken', arguments: '{}' }],
      modelResponseInfo: {
        responseId: 'resp-tool',
        responseModel: 'gpt-test',
        stopReason: ModelStopReasons.TOOL_CALL,
        usage: UsageStats.empty(),
      },
    };

    const finalResponse = {
      messageId: 'msg-2',
      type: 'response_api_message',
      role: 'assistant',
      content: [{ type: 'text', text: 'done' }],
      modelResponseInfo: {
        responseId: 'resp-final',
        responseModel: 'gpt-test',
        stopReason: ModelStopReasons.STOP,
        usage: UsageStats.empty(),
      },
    };

    const dependencies = createDependencies();
    const executeMock = vi.fn().mockResolvedValue(toolResult);
    dependencies.toolRegistry.execute = executeMock;

    const transport = createTransport(async () => toolResponse as unknown as RawMessageType);
    const generateMock = transport.generate as unknown as ReturnType<typeof vi.fn>;
    generateMock
      .mockResolvedValueOnce(toolResponse as unknown as RawMessageType)
      .mockResolvedValueOnce(finalResponse as unknown as RawMessageType);

    const { loop, getTransportSpy } = createLoop(dependencies, transport);
    const modelOptions: ModelOptions = {
      model: 'gpt-test',
      apiMode: ApiModes.RESPONSE,
      parallelToolCalls: false,
    };

    try {
      const result = await loop.execute(modelOptions, { role: 'user', content: 'hi' } as never, new AbortController().signal);

      expect(result.action).toBe('execution');
      if (result.action === 'execution') {
        expect(result.stopReason).toBe('end_turn');
      }
      expect(executeMock).toHaveBeenCalledTimes(1);
      expect(generateMock).toHaveBeenCalledTimes(2);
    } finally {
      getTransportSpy.mockRestore();
    }
  });
});
