import { describe, expect, it } from 'vitest';
import { AnthropicTransport } from './anthropic';
import { ApiModes, type AnthropicApiMessage, type SessionMessage } from '../../message';

describe('AnthropicTransport conversion', () => {
  const transport = new AnthropicTransport(undefined, 'test-api-key');

  it('converts raw anthropic tool_result blocks into a tool session message', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'anthro-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'text', text: 'assistant answer' },
          {
            type: 'tool_use',
            id: 'tool-1',
            name: 'search_tool',
            input: {
              query: 'pixi',
            },
          },
          {
            type: 'tool_result',
            tool_use_id: 'call-1',
            content: 'search result',
          },
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);

    expect(session.messageId).toBe('anthro-1');
    expect(session.type).toBe('session_message');
    expect(session.role).toBe('assistant');
    expect(Array.isArray(session.content)).toBe(true);
    expect(session.content).toMatchObject([
      { type: 'text', text: 'assistant answer' },
      {
        type: 'tool_call',
        id: 'tool-1',
        name: 'search_tool',
        arguments: '{"query":"pixi"}',
      },
      {
        type: 'tool_result',
        id: 'call-1',
        result: 'search result',
      },
    ]);
  });

  it('converts assistant session tool_call content into anthropic tool_use raw content', () => {
    const sessionMessage: SessionMessage = {
      messageId: 'anthro-2',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: 'call-2',
          name: 'lookup',
          arguments: '{"symbol":"TEST"}',
        },
      ],
    };

    const raw = transport.convertToRawMessage(sessionMessage);

    expect(raw).toEqual({
      messageId: 'anthro-2',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: 'call-2',
            name: 'lookup',
            input: {
              symbol: 'TEST',
            },
          },
        ],
      },
      metadata: undefined,
    });
  });

  it('converts user string session message into anthropic raw text content', () => {
    const sessionMessage: SessionMessage = {
      messageId: 'anthro-3',
      type: 'session_message',
      role: 'user',
      content: 'hello anthropic',
    };

    const raw = transport.convertToRawMessage(sessionMessage);

    expect(raw).toEqual({
      messageId: 'anthro-3',
      type: 'anthropic_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: 'hello anthropic',
      },
      metadata: undefined,
    });
  });

  it('converts raw anthropic server_tool_use blocks into session tool_call', () => {
    const raw = {
      messageId: 'anthro-4',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'server_tool_use',
            id: 'server-1',
            name: 'custom-tool',
            input: { test: true },
          },
        ],
      },
    } as unknown as AnthropicApiMessage;

    const session = transport.convertFromRawMessage(raw);

    expect(session.content).toMatchObject([
      {
        type: 'server_tool_use',
        name: 'custom-tool',
        providerSpecific: ApiModes.ANTHROPIC,
        data: JSON.stringify({ id: 'server-1', input: { test: true } }),
      },
    ]);
  });
});
