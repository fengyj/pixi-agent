import { describe, expect, it } from 'vitest';
import { ChatCompletionMessageConverter } from './chat_completion.converters';
import type { ChatCompletionApiMessage, SessionMessage } from '../../message';

describe('ChatCompletionMessageConverter', () => {
  const converter = new ChatCompletionMessageConverter();

  it('converts assistant raw content into session parts', () => {
    const raw = {
      messageId: 'msg-1',
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
    } as ChatCompletionApiMessage;

    const session = converter.convertFromRawMessage(raw);

    expect(session.role).toBe('assistant');
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
      { type: 'refusal', reason: 'policy' },
    ]);
  });

  it('converts session assistant content into raw chat completion message', () => {
    const session: SessionMessage = {
      messageId: 'msg-2',
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

    const raw = converter.convertToRawMessage(session) as ChatCompletionApiMessage;

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
});
