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

  // ── round-trip tests ─────────────────────────────────────────────────────

  it('round-trips thinking blocks', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'rt-think',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'step by step...', signature: 'sig123' },
          { type: 'text', text: 'final answer' },
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([
      { type: 'thinking', content: 'step by step...', signature: 'sig123' },
      { type: 'text', text: 'final answer' },
    ]);

    const back = transport.convertToRawMessage(session as SessionMessage);
    expect((back as AnthropicApiMessage).content).toMatchObject({
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'step by step...', signature: 'sig123' },
        { type: 'text', text: 'final answer' },
      ],
    });
  });

  it('round-trips web_search_tool_result', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'rt-ws',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'web_search_tool_result',
            tool_use_id: 'ws-1',
            query: 'pixi',
            results: [{ title: 'Pixi', url: 'https://pixi.sh' }],
          } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    const parts = session.content as Array<Record<string, unknown>>;
    expect(parts[0]).toMatchObject({
      type: 'tool_result',
      id: 'ws-1',
      name: 'web_search',
      providerSpecific: ApiModes.ANTHROPIC,
    });

    // Round-trip
    const back = transport.convertToRawMessage(session as SessionMessage);
    const backContent = (back as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'web_search_tool_result',
      tool_use_id: 'ws-1',
    });
  });

  it('round-trips document block with text source', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'rt-doc',
      type: 'anthropic_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          {
            type: 'document',
            source: { type: 'text', media_type: 'text/plain', data: 'file contents' },
          } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    // text source documents become TextPart arrays
    expect(session.content).toMatchObject([{ type: 'text', text: 'file contents' }]);
  });

  it('round-trips tool_use -> session -> tool_use (with empty input)', () => {
    const session: SessionMessage = {
      messageId: 'rt-empty',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'tool_call', id: 'call-e', name: 'flag', arguments: '' },
      ],
    };

    const raw = transport.convertToRawMessage(session);
    const backOuter = (raw as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backOuter.content[0]).toMatchObject({
      type: 'tool_use',
      id: 'call-e',
      name: 'flag',
      input: null,
    });
  });

  // ── specific block type conversions ──────────────────────────────────────

  it('converts search_result blocks to text parts', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'sr-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'search_result',
            source: 'web',
            title: 'Result Title',
            content: 'result body',
          } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([
      {
        type: 'text',
        text: JSON.stringify({
          source: 'web',
          title: 'Result Title',
          content: 'result body',
          type: 'search_result',
        }),
      },
    ]);
  });

  it('converts tool_reference blocks to text parts', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'tr-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'tool_reference', tool_name: 'search' } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([
      {
        type: 'text',
        text: JSON.stringify({ tool_name: 'search', type: 'tool_reference' }),
      },
    ]);
  });

  it('drops redacted_thinking blocks', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'red-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'redacted_thinking', data: '...' } as never,
          { type: 'text', text: 'visible' },
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([{ type: 'text', text: 'visible' }]);
  });

  it('converts container_upload to document part', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'cu-1',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          { type: 'container_upload', file_id: 'file-abc' } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([
      {
        type: 'document',
        document: { sourceType: 'file_id', fileId: 'file-abc' },
      },
    ]);
  });

  it('converts image block with base64 source', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'img-b64',
      type: 'anthropic_api_message',
      role: 'user',
      content: {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/png', data: 'abc123' },
          } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    expect(session.content).toMatchObject([
      {
        type: 'image',
        image: { sourceType: 'base64', mimeType: 'image/png', data: 'abc123' },
      },
    ]);

    // Round-trip
    const back = transport.convertToRawMessage(session as SessionMessage);
    const backContent = (back as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'image',
      source: { type: 'base64', media_type: 'image/png', data: 'abc123' },
    });
  });

  // ── citation handling ────────────────────────────────────────────────────

  it('preserves web_search_result_location citations in round-trip', () => {
    const raw: AnthropicApiMessage = {
      messageId: 'cit-web',
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: [
          {
            type: 'text',
            text: 'cited answer',
            citations: [
              {
                type: 'web_search_result_location',
                cited_text: 'Pixi',
                title: 'Pixi Docs',
                url: 'https://example.com',
              } as never,
            ],
          } as never,
        ],
      },
    };

    const session = transport.convertFromRawMessage(raw);
    const back = transport.convertToRawMessage(session as SessionMessage);
    const backContent = (back as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'text',
      text: 'cited answer',
      citations: [
        {
          type: 'web_search_result_location',
          cited_text: 'Pixi',
          title: 'Pixi Docs',
          url: 'https://example.com',
        },
      ],
    });
  });

  // ── provider-specific server_tool_use round-trip ─────────────────────────

  it('round-trips server_tool_use with providerSpecific mark', () => {
    const session: SessionMessage = {
      messageId: 'stu-rt',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'server_tool_use',
          name: 'custom_tool',
          data: JSON.stringify({ id: 'stu-1', input: { key: 'val' } }),
          providerSpecific: ApiModes.ANTHROPIC,
        },
      ],
    };

    const raw = transport.convertToRawMessage(session);
    const backContent = (raw as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      name: 'custom_tool',
      id: 'stu-1',
      input: { key: 'val' },
    });
  });

  // ── fallback for non-Anthropic server_tool_use ───────────────────────────

  it('falls back to text block for non-Anthropic server_tool_use', () => {
    const session: SessionMessage = {
      messageId: 'stu-fb',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'server_tool_use',
          name: 'mcp_tool',
          data: '{"op":"read"}',
          providerSpecific: ApiModes.RESPONSE,
        },
      ],
    };

    const raw = transport.convertToRawMessage(session);
    const backContent = (raw as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'text',
      text: 'Tool use: mcp_tool with data {"op":"read"}',
    });
  });

  // ── media fallback to text ──────────────────────────────────────────────

  it('demotes image with file_id to text fallback', () => {
    const session: SessionMessage = {
      messageId: 'img-fb',
      type: 'session_message',
      role: 'user',
      content: [
        {
          type: 'image',
          image: { sourceType: 'file_id', fileId: 'f-123' },
        },
      ],
    };

    const raw = transport.convertToRawMessage(session);
    const backContent = (raw as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'text',
    });
    expect(backContent.content[0].text).toContain('"fileId":"f-123"');
  });

  it('demotes document with file_id to text fallback', () => {
    const session: SessionMessage = {
      messageId: 'doc-fb',
      type: 'session_message',
      role: 'user',
      content: [
        {
          type: 'document',
          document: { sourceType: 'file_id', fileId: 'f-doc' },
        },
      ],
    };

    const raw = transport.convertToRawMessage(session);
    const backContent = (raw as AnthropicApiMessage).content as {
      role: string;
      content: Array<Record<string, unknown>>;
    };
    expect(backContent.content[0]).toMatchObject({
      type: 'text',
    });
    // Should contain the source doc data in the JSON fallback
    expect(backContent.content[0].text).toContain('"fileId":"f-doc"');
  });
});
