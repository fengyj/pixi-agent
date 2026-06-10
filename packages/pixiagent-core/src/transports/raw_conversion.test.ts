import { describe, expect, it } from 'vitest';
import { ChatCompletionTransport } from './chat-completion/chat_completion';
import { AnthropicTransport } from './anthropic/anthropic';
import { ResponseTransport } from './response/response';
import type {
  AnthropicApiMessage,
  ChatCompletionApiMessage,
  ResponseApiMessage,
  SessionMessage,
} from '../message';

describe('raw message cross-transport conversion via SessionMessage', () => {
  const chatTransport = new ChatCompletionTransport(undefined, 'test-api-key');
  const anthropicTransport = new AnthropicTransport(undefined, 'test-api-key');
  const responseTransport = new ResponseTransport(undefined, 'test-api-key');

  it('converts ChatCompletion assistant function_call -> SessionMessage -> Anthropic tool_use', () => {
    const rawChat: ChatCompletionApiMessage = {
      messageId: 'chat-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        function_call: {
          name: 'weather_lookup',
          arguments: '{"location":"Seattle"}',
        },
      },
    };

    const session = chatTransport.convertFromRawMessage(rawChat);
    expect(session.role).toBe('assistant');
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<unknown>)[0]).toMatchObject({
      type: 'tool_call',
      name: 'weather_lookup',
      arguments: '{"location":"Seattle"}',
    });

    const anthroRaw = anthropicTransport.convertToRawMessage(session as SessionMessage);
    expect(anthroRaw).toEqual({
      messageId: 'chat-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: '',
            name: 'weather_lookup',
            input: {
              location: 'Seattle',
            },
          },
        ],
      },
      metadata: undefined,
    });
  });

  it('converts Anthropic tool_result -> SessionMessage -> Response function_call_output', () => {
    const rawAnthropic: AnthropicApiMessage = {
      messageId: 'anthro-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'call-42',
            content: 'search results here',
          },
        ],
      },
    };

    const session = anthropicTransport.convertFromRawMessage(rawAnthropic);
    // Session preserves the raw message's role (assistant), regardless of content block types
    expect(session.role).toBe('assistant');
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<unknown>)[0]).toMatchObject({
      type: 'tool_result',
      id: 'call-42',
      result: 'search results here',
    });

    // Change role to 'tool' for the cross-SDK conversion
    const toolSession = { ...session, role: 'tool' as const };
    const responseRaw = responseTransport.convertToRawMessage(toolSession as SessionMessage);
    expect(responseRaw).toEqual({
      messageId: 'anthro-1',
      type: 'response_api_message',
      role: 'tool',
      metadata: undefined,
      content: [
        {
          type: 'function_call_output',
          call_id: 'call-42',
          output: 'search results here',
          status: 'completed',
        },
      ],
    });
  });

  it('converts Response assistant output_text -> SessionMessage -> ChatCompletion text', () => {
    const rawResponse: ResponseApiMessage = {
      messageId: 'resp-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'message',
          id: 'resp-1',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'Hello from Response', annotations: [] }],
        },
      ],
    };

    const session = responseTransport.convertFromRawMessage(rawResponse);
    expect(session.role).toBe('assistant');
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<unknown>)[0]).toMatchObject({
      type: 'text',
      text: 'Hello from Response',
    });

    const chatRaw = chatTransport.convertToRawMessage(session as SessionMessage);
    // ChatCompletion simplifies a single text part back to a plain string
    expect(chatRaw).toMatchObject({
      messageId: 'resp-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: 'Hello from Response',
        audio: null,
      },
      metadata: undefined,
    });
  });

  // ── thinking / reasoning cross-SDK ──────────────────────────────────────

  it('converts Anthropic thinking -> SessionMessage -> Response reasoning', () => {
    const rawAnthropic: AnthropicApiMessage = {
      messageId: 'think-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'step by step', signature: 'sig1' },
          { type: 'text', text: 'final' },
        ],
      },
    };

    const session = anthropicTransport.convertFromRawMessage(rawAnthropic);
    expect(session.content).toMatchObject([
      { type: 'thinking', content: 'step by step', signature: 'sig1' },
      { type: 'text', text: 'final' },
    ]);

    const responseRaw = responseTransport.convertToRawMessage(session as SessionMessage);
    expect(responseRaw.content).toMatchObject([
      { type: 'reasoning', summary: [{ type: 'summary_text', text: 'step by step' }] },
      {
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text: 'final', annotations: [] }],
      },
    ]);
  });

  it('converts Response reasoning -> SessionMessage -> Anthropic thinking', () => {
    const rawResponse: ResponseApiMessage = {
      messageId: 'reason-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'reasoning',
          id: 'r1',
          summary: [{ type: 'summary_text', text: 'I should search' }],
        },
        {
          type: 'message',
          id: 'm1',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'result', annotations: [] }],
        },
      ],
    };

    const session = responseTransport.convertFromRawMessage(rawResponse);
    expect(session.content).toMatchObject([
      { type: 'thinking', content: 'I should search' },
      { type: 'text', text: 'result' },
    ]);

    const anthroRaw = anthropicTransport.convertToRawMessage(session as SessionMessage);
    expect((anthroRaw as AnthropicApiMessage).content).toMatchObject({
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'I should search', signature: '' },
        { type: 'text', text: 'result' },
      ],
    });
  });

  // ── Anthropic tool_use -> ChatCompletion tool_calls ──────────────────────

  it('converts Anthropic tool_use -> SessionMessage -> ChatCompletion tool_calls', () => {
    const rawAnthropic: AnthropicApiMessage = {
      messageId: 'tool-x',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'tu-1',
            name: 'get_weather',
            input: { city: 'Seattle' },
          },
        ],
      },
    };

    const session = anthropicTransport.convertFromRawMessage(rawAnthropic);
    const chatRaw = chatTransport.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    expect(chatRaw.content).toMatchObject({
      role: 'assistant',
      tool_calls: [
        {
          type: 'function',
          id: 'tu-1',
          function: { name: 'get_weather', arguments: '{"city":"Seattle"}' },
        },
      ],
    });
  });

  // ── ChatCompletion tool_calls -> Response function_call ──────────────────

  it('converts ChatCompletion tool_calls -> SessionMessage -> Response function_call', () => {
    const rawChat: ChatCompletionApiMessage = {
      messageId: 'cc-x',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            type: 'function',
            id: 'c-1',
            function: { name: 'search', arguments: '{"q":"test"}' },
          },
        ],
      },
    };

    const session = chatTransport.convertFromRawMessage(rawChat);
    const responseRaw = responseTransport.convertToRawMessage(session as SessionMessage);

    expect(responseRaw.content).toMatchObject([
      {
        type: 'function_call',
        call_id: 'c-1',
        name: 'search',
        arguments: '{"q":"test"}',
        status: 'completed',
      },
    ]);
  });

  // ── Anthropic specific tool result -> ChatCompletion tool message ────────

  it('converts Anthropic web_search_tool_result -> SessionMessage -> ChatCompletion tool message', () => {
    const rawAnthropic: AnthropicApiMessage = {
      messageId: 'ws-x',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'web_search_tool_result',
            tool_use_id: 'ws-99',
            query: 'pixi',
            results: [],
          } as never,
        ],
      },
    };

    const session = anthropicTransport.convertFromRawMessage(rawAnthropic);
    // Session preserves raw message role – tool_result blocks from an assistant message don't change the role
    expect(session.role).toBe('assistant');

    // Use role 'tool' for cross-SDK tool message conversion
    const toolSession = { ...session, role: 'tool' as const };
    const chatRaw = chatTransport.convertToRawMessage(toolSession as SessionMessage) as ChatCompletionApiMessage;
    expect(chatRaw.content).toMatchObject({
      role: 'tool',
      tool_call_id: 'ws-99',
    });
  });

  // ── Response function_call_output -> Anthropic tool_result ───────────────

  it('converts Response function_call_output -> SessionMessage -> Anthropic tool_result', () => {
    const rawResponse: ResponseApiMessage = {
      messageId: 'fo-x',
      type: 'response_api_message',
      role: 'tool',
      content: [
        {
          type: 'function_call_output',
          call_id: 'fc-1',
          output: '{"data": 42}',
          status: 'completed',
        },
      ],
    };

    const session = responseTransport.convertFromRawMessage(rawResponse);
    const anthroRaw = anthropicTransport.convertToRawMessage(session as SessionMessage);

    expect((anthroRaw as AnthropicApiMessage).content).toMatchObject({
      role: 'user',
      content: [
        {
          type: 'tool_result',
          tool_use_id: 'fc-1',
          content: '{"data": 42}',
        },
      ],
    });
  });

  // ── server_tool_use cross-SDK: Response -> ChatCompletion (dropped) ─────

  it('drops server_tool_use parts when converting Response -> ChatCompletion (expected)', () => {
    const session: SessionMessage = {
      messageId: 'stu-drop',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'server_tool_use',
          name: 'mcp_call',
          data: '{"server_label":"gh"}',
          providerSpecific: 'response',
        },
        { type: 'text', text: 'done' },
      ],
    };

    const chatRaw = chatTransport.convertToRawMessage(session) as ChatCompletionApiMessage;
    // server_tool_use is demoted to text, not dropped
    expect((chatRaw.content as { role: string; content: Array<Record<string, unknown>> }).content).toMatchObject([
      { type: 'text', text: 'Tool use: mcp_call with data {"server_label":"gh"}' },
      { type: 'text', text: 'done' },
    ]);
  });

  // ── server_tool_use cross-SDK: Anthropic -> Response (falls back to text) ─

  it('falls back to text for Anthropic server_tool_use when converting to Response', () => {
    const session: SessionMessage = {
      messageId: 'stu-fb-x',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'server_tool_use',
          name: 'custom_tool',
          data: '{"id":"s1","input":{"k":"v"}}',
          providerSpecific: 'anthropic',
        },
      ],
    };

    const responseRaw = responseTransport.convertToRawMessage(session);
    // Response wraps the fallback text inside a message envelope
    expect(responseRaw.content).toMatchObject([
      {
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'output_text',
            text: 'Tool use: custom_tool with data {"id":"s1","input":{"k":"v"}}',
          },
        ],
      },
    ]);
  });
});
