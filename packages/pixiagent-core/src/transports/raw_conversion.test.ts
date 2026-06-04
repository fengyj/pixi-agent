import { describe, expect, it } from 'vitest';
import { ChatCompletionTransport } from './chat_completion';
import { AnthropicTransport } from './anthropic';
import { ResponseTransport } from './response';
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
    expect(session.role).toBe('tool');
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<unknown>)[0]).toMatchObject({
      type: 'tool_result',
      id: 'call-42',
      result: 'search results here',
    });

    const responseRaw = responseTransport.convertToRawMessage(session as SessionMessage);
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
    expect(chatRaw).toMatchObject({
      messageId: 'resp-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [{ type: 'text', text: 'Hello from Response' }],
        audio: null,
      },
      metadata: undefined,
    });
  });
});
