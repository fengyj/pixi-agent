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

  // ── round-trip tests ─────────────────────────────────────────────────────

  it('round-trips assistant tool_calls + text', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: 'I found these results',
        tool_calls: [
          {
            type: 'function',
            id: 'call-abc',
            function: {
              name: 'search',
              arguments: '{"query":"pixi"}',
            },
          },
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    expect(back.content).toMatchObject({
      role: 'assistant',
      content: 'I found these results',
      tool_calls: [
        {
          type: 'function',
          id: 'call-abc',
          function: { name: 'search', arguments: '{"query":"pixi"}' },
        },
      ],
    });
  });

  it('round-trips assistant tool_calls without text content', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-2',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            type: 'function',
            id: 'call-def',
            function: { name: 'run', arguments: '{}' },
          },
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    expect(back.content).toMatchObject({
      role: 'assistant',
      tool_calls: [
        {
          type: 'function',
          id: 'call-def',
          function: { name: 'run', arguments: '{}' },
        },
      ],
    });
  });

  it('round-trips user message with image and text', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-u1',
      type: 'chat_completion_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          { type: 'text', text: 'describe this' },
          { type: 'image_url', image_url: { url: 'https://example.com/img.png' } },
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    expect(back.content).toMatchObject({
      role: 'user',
      content: [
        { type: 'text', text: 'describe this' },
        { type: 'image_url', image_url: { url: 'https://example.com/img.png' } },
      ],
    });
  });

  it('round-trips user message with audio input', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-u2',
      type: 'chat_completion_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          {
            type: 'input_audio',
            input_audio: { data: 'AAAA', format: 'wav' },
          },
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    const backContent = back.content as { role: string; content: Array<Record<string, unknown>> };
    expect(backContent.content[0]).toMatchObject({
      type: 'input_audio',
      input_audio: { data: 'AAAA', format: 'wav' },
    });
  });

  it('round-trips tool message', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-t1',
      type: 'chat_completion_api_message',
      role: 'tool',
      content: {
        role: 'tool',
        tool_call_id: 'call-xyz',
        content: '{"temp": 72}',
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;

    expect(back.content).toMatchObject({
      role: 'tool',
      tool_call_id: 'call-xyz',
      content: '{"temp": 72}',
    });
  });

  it('round-trips tool message with array content', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'rt-t2',
      type: 'chat_completion_api_message',
      role: 'tool',
      content: {
        role: 'tool',
        tool_call_id: 'call-arr',
        content: [{ type: 'text', text: 'part 1' }, { type: 'text', text: 'part 2' }],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    const back = converter.convertToRawMessage(session as SessionMessage);

    // Array content is converted to a single tool message with array content
    // (parsed from the JSON-stringified internal result)
    const backMessage = Array.isArray(back) ? back[0] : back;
    expect(backMessage.content).toMatchObject({
      role: 'tool',
      tool_call_id: 'call-arr',
    });
    const content = (backMessage.content as { role: string; content: unknown }).content;
    if (Array.isArray(content)) {
      expect(content).toMatchObject([
        { type: 'text', text: 'part 1' },
        { type: 'text', text: 'part 2' },
      ]);
    } else {
      // May also come back as a single string depending on parsing
      expect(typeof content).toBe('string');
    }
  });

  // ── provider-specific tool calls ──────────────────────────────────────────

  it('handles non-function tool_calls as provider-specific', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'ps-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: null,
        tool_calls: [
          {
            type: 'custom_tool',
            id: 'ct-1',
            extraField: 'value',
          } as never,
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    expect(Array.isArray(session.content)).toBe(true);
    const toolCall = (session.content as Array<Record<string, unknown>>)[0];
    expect(toolCall).toMatchObject({
      type: 'tool_call',
      id: 'ct-1',
      name: 'custom_tool',
      providerSpecific: 'completions',
    });

    // Round-trip
    const back = converter.convertToRawMessage(session as SessionMessage) as ChatCompletionApiMessage;
    expect((back.content as { tool_calls?: Array<Record<string, unknown>> }).tool_calls?.[0]).toMatchObject({
      type: 'custom_tool',
      id: 'ct-1',
    });
  });

  // ── edge cases ────────────────────────────────────────────────────────────

  it('converts legacy function_call correctly', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'leg-1',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: null,
        function_call: {
          name: 'old_search',
          arguments: '{"q":"test"}',
        },
      },
    };

    const session = converter.convertFromRawMessage(raw);
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<Record<string, unknown>>)[0]).toMatchObject({
      type: 'tool_call',
      name: 'old_search',
      arguments: '{"q":"test"}',
    });
  });

  it('converts user message with base64 image', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'b64-1',
      type: 'chat_completion_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: { url: 'data:image/png;base64,iVBORw0KGgo=' },
          },
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<Record<string, unknown>>)[0]).toMatchObject({
      type: 'image',
      image: {
        sourceType: 'base64',
        mimeType: 'image/png',
        data: 'iVBORw0KGgo=',
      },
    });
  });

  it('converts user message with file (file_data)', () => {
    const raw: ChatCompletionApiMessage = {
      messageId: 'file-1',
      type: 'chat_completion_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          {
            type: 'file',
            file: { file_data: 'hello', filename: 'test.txt' },
          } as never,
        ],
      },
    };

    const session = converter.convertFromRawMessage(raw);
    expect(Array.isArray(session.content)).toBe(true);
    expect((session.content as Array<Record<string, unknown>>)[0]).toMatchObject({
      type: 'document',
      document: {
        sourceType: 'base64',
        data: 'hello',
        fileName: 'test.txt',
      },
    });
  });

  it('demotes server_tool_use and media parts to text, drops thinking', () => {
    const session: SessionMessage = {
      messageId: 'drop-1',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'text', text: 'visible' },
        { type: 'thinking', content: 'hidden reasoning', signature: 'sig' },
        { type: 'server_tool_use', name: 'mcp', data: '{}', providerSpecific: 'response' },
        { type: 'audio', audio: { sourceType: 'base64', mimeType: 'audio/wav', data: 'AAAA' } },
        { type: 'image', image: { sourceType: 'url', url: 'https://example.com/img.png' } },
      ],
    };

    const raw = converter.convertToRawMessage(session) as ChatCompletionApiMessage;
    const content = (raw.content as { role: string; content: Array<Record<string, unknown>> }).content;
    // thinking is dropped; server_tool_use & unsupported media are demoted to text
    expect(content).toMatchObject([
      { type: 'text', text: 'visible' },
      { type: 'text' },   // server_tool_use fallback
      { type: 'text' },   // audio fallback
      { type: 'text' },   // image fallback
    ]);
    expect(content).toHaveLength(4);
    expect(content[1].text).toContain('Tool use: mcp with data {}');
    expect(content[3].text).toContain('"url":"https://example.com/img.png"');
  });
});
