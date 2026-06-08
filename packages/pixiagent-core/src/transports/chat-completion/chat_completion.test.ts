import { describe, expect, it } from 'vitest';
import { ChatCompletionTransport } from './chat_completion';
import type { ChatCompletionApiMessage, SessionMessage } from '../../message';

describe('ChatCompletionTransport conversion', () => {
  const transport = new ChatCompletionTransport(undefined, 'test-api-key');

  it('converts assistant string raw message to session message', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'msg-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: 'hello world',
        name: 'assistant-1',
      },
      metadata: { source: 'unit-test' },
    };

    const session = transport.convertFromRawMessage(raw);

    expect(session).toEqual({
      messageId: 'msg-1',
      type: 'session_message',
      role: 'assistant',
      content: 'hello world',
      name: 'assistant-1',
      metadata: { source: 'unit-test' },
    });
  });

  it('attaches citations to the first text part when raw assistant content includes annotations', () => {
    const raw = {
      messageId: 'msg-2',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: 'hello from citation',
        refusal: 'policy',
        annotations: [
          {
            type: 'url_citation',
            url_citation: {
              url: 'https://example.com',
              title: 'Example',
              start_index: 0,
              end_index: 5,
            },
          },
        ],
      },
    } as ChatCompletionApiMessage & { content: { annotations: Array<unknown> } };

    const session = transport.convertFromRawMessage(raw);

    expect(session.type).toBe('session_message');
    expect(Array.isArray(session.content)).toBe(true);
    expect(session.content).toEqual([
      {
        type: 'text',
        text: 'hello from citation',
        citations: [
          {
            type: 'web_location',
            url: 'https://example.com',
            citedText: '',
            title: 'Example',
            startIndex: 0,
            endIndex: 5,
            extra: { rawCitationType: 'url_citation' },
          },
        ],
      },
      {
        type: 'refusal',
        reason: 'policy',
      },
    ]);
  });

  it('converts assistant function_call raw message to a session tool_call', () => {
    const raw = {
      messageId: 'msg-5',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        function_call: {
          name: 'doThing',
          arguments: '{"foo":"bar"}',
        },
      },
    } as ChatCompletionApiMessage;

    const session = transport.convertFromRawMessage(raw);

    expect(session).toEqual({
      messageId: 'msg-5',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: '',
          name: 'doThing',
          arguments: '{"foo":"bar"}',
        },
      ],
      metadata: undefined,
    });
  });

  it('converts assistant tool_call plus text session content into raw chat_completion tool_calls', () => {
    const session: SessionMessage = {
      messageId: 'msg-6',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: 'call-4',
          name: 'search',
          arguments: '{"query":"pixi"}',
        },
        { type: 'text', text: 'result available' },
      ],
    };

    const raw = transport.convertToRawMessage(session) as ChatCompletionApiMessage;

    expect(raw.content).toMatchObject({
      role: 'assistant',
      content: 'result available',
      tool_calls: [
        {
          type: 'function',
          id: 'call-4',
          function: {
            name: 'search',
            arguments: '{"query":"pixi"}',
          },
        },
      ],
    });
  });

  it('converts assistant refusal-only session message into raw refusal payload', () => {
    const session: SessionMessage = {
      messageId: 'msg-3',
      type: 'session_message',
      role: 'assistant',
      content: [{ type: 'refusal', reason: 'blocked by policy' }],
    };

    const raw = transport.convertToRawMessage(session);

    expect(Array.isArray(raw)).toBe(false);
    expect((raw as ChatCompletionApiMessage).content).toMatchObject({
      role: 'assistant',
      content: null,
      refusal: 'blocked by policy',
    });
  });

  it('converts user image part into raw image_url content', () => {
    const session = {
      messageId: 'msg-4',
      type: 'session_message',
      role: 'user',
      content: [
        {
          type: 'image',
          image: {
            sourceType: 'url',
            url: 'https://cdn.example.com/pic.png',
          },
        },
      ],
    } as SessionMessage;

    const raw = transport.convertToRawMessage(session) as ChatCompletionApiMessage;

    expect(raw.content).toEqual({
      role: 'user',
      content: [
        {
          type: 'image_url',
          image_url: {
            url: 'https://cdn.example.com/pic.png',
          },
        },
      ],
    });
  });
});
