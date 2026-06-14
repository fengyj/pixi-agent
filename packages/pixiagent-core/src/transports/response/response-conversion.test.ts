import { describe, expect, it } from 'vitest';
import { ResponseConversionHelper } from './response-conversion';
import type {
  ResponseComputerToolCallOutputItem,
  ResponseCustomToolCall,
  ResponseFileSearchToolCall,
  ResponseFunctionWebSearch,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseReasoningItem,
} from 'openai/resources/responses/responses';
import { ApiModes } from '../../message';

describe('ResponseConversionHelper', () => {
  it('maps response input message items to content parts', () => {
    const item: ResponseInputItem.Message = {
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text: 'hello world' }],
    };

    expect(ResponseConversionHelper.toContentParts(item)).toEqual([
      { type: 'text', text: 'hello world' },
    ]);
  });

  it('maps response function_call items to tool_call content parts', () => {
    const item: ResponseOutputItem = {
      type: 'function_call',
      id: 'func-1',
      call_id: 'call-1',
      name: 'search',
      arguments: '{"q":"pixi"}',
      status: 'completed',
    };

    expect(ResponseConversionHelper.toContentParts(item)).toEqual([
      {
        type: 'tool_call',
        id: 'call-1',
        name: 'search',
        arguments: '{"q":"pixi"}',
      },
    ]);
  });

  it('builds response items from session content for assistant role', () => {
    const items = ResponseConversionHelper.toResponseItems('assistant', [
      { type: 'tool_call', id: 'tool-1', name: 'run', arguments: '{"cmd":"ls"}' },
      { type: 'text', text: 'ok' },
    ]);

    expect(items).toMatchObject([
      {
        type: 'function_call',
        id: 'tool-1',
        call_id: 'tool-1',
        name: 'run',
        arguments: '{"cmd":"ls"}',
        status: 'completed',
      },
      {
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text: 'ok', annotations: [] }],
      },
    ]);
  });

  // ── round-trip tests ─────────────────────────────────────────────────────

  it('round-trips reasoning items', () => {
    const item: ResponseReasoningItem = {
      type: 'reasoning',
      id: 'reason-1',
      summary: [{ type: 'summary_text', text: 'I should search first' }],
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts).toMatchObject([{ type: 'thinking', content: 'I should search first' }]);

    // Session → Response
    const items = ResponseConversionHelper.toResponseItems('assistant', [
      { type: 'thinking', content: 'I should search first' },
    ]);
    expect(items).toMatchObject([
      {
        type: 'reasoning',
        summary: [{ type: 'summary_text', text: 'I should search first' }],
      },
    ]);
  });

  it('round-trips image_generation_call', () => {
    const item: ResponseOutputItem.ImageGenerationCall = {
      type: 'image_generation_call',
      id: 'img-1',
      result: 'base64data',
      status: 'completed',
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts).toMatchObject([
      {
        type: 'image',
        image: { sourceType: 'base64', mimeType: 'image/png', data: 'base64data' },
      },
    ]);

    // Session → Response
    const items = ResponseConversionHelper.toResponseItems('assistant', [
      {
        type: 'image',
        image: { sourceType: 'base64', mimeType: 'image/png', data: 'newdata' },
      },
    ]);
    expect(items).toMatchObject([
      { type: 'image_generation_call', result: 'newdata', status: 'completed' },
    ]);
  });

  it('round-trips custom_tool_call with call_id', () => {
    const item: ResponseCustomToolCall = {
      type: 'custom_tool_call',
      id: 'ct-1',
      call_id: 'call-1',
      input: 'val',
    } as ResponseCustomToolCall;

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'tool_call',
      id: 'call-1',
      name: 'custom_tool_call',
      providerSpecific: 'response',
    });

    // Round-trip via SessionMessage
    const sessionMsg = {
      messageId: 'rt-ct',
      type: 'session_message' as const,
      role: 'assistant' as const,
      content: parts,
    };
    const items = ResponseConversionHelper.toResponseOutputItems(sessionMsg);
    expect(items[0]).toMatchObject({
      type: 'custom_tool_call',
      call_id: 'call-1',
      input: 'val',
    });
  });

  it('round-trips computer_call_output', () => {
    const item: ResponseComputerToolCallOutputItem = {
      type: 'computer_call_output',
      id: 'cco-1',
      call_id: 'comp-1',
      output: { type: 'computer_screenshot', file_id: 'screenshot result' },
      status: 'completed',
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'tool_result',
      id: 'comp-1',
      name: 'computer_call_output',
      result: JSON.stringify({
        id: 'cco-1',
        output: { type: 'computer_screenshot', file_id: 'screenshot result' },
        status: 'completed',
      }),
      providerSpecific: 'response',
    });
  });

  // ── server_tool_use (no call_id) conversions ─────────────────────────────

  it('converts mcp_call to server_tool_use', () => {
    const item = {
      type: 'mcp_call',
      id: 'mcp-1',
      name: 'mcp_call',
      server_label: 'github',
      tool_name: 'search_issues',
      arguments: '{"q":"bug"}',
      status: 'completed',
    } as ResponseOutputItem.McpCall;

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'server_tool_use',
      name: 'mcp_call',
      providerSpecific: 'response',
    });

    // Round-trip
    const sessionMsg = {
      messageId: 'rt-mcp',
      type: 'session_message' as const,
      role: 'assistant' as const,
      content: parts,
    };
    const items = ResponseConversionHelper.toResponseOutputItems(sessionMsg);
    expect(items[0]).toMatchObject({
      type: 'mcp_call',
      server_label: 'github',
      tool_name: 'search_issues',
    });
  });

  it('converts file_search_call to server_tool_use', () => {
    const item = {
      type: 'file_search_call',
      id: 'fs-1',
      queries: ['search term'],
      status: 'completed',
    } as ResponseFileSearchToolCall;

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'server_tool_use',
      name: 'file_search_call',
      providerSpecific: 'response',
    });
  });

  it('converts web_search_call to server_tool_use', () => {
    const item = {
      type: 'web_search_call',
      id: 'ws-1',
      action: { type: 'search', query: 'search query' },
      status: 'completed',
    } as ResponseFunctionWebSearch;

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'server_tool_use',
      name: 'web_search_call',
      providerSpecific: 'response',
    });
  });

  // ── dropped items ────────────────────────────────────────────────────────

  it('drops compaction and item_reference', () => {
    expect(ResponseConversionHelper.toContentParts({ type: 'compaction', id: 'c1' } as never)).toEqual([]);
    expect(ResponseConversionHelper.toContentParts({ type: 'compaction_trigger', id: 'ct1' } as never)).toEqual([]);
    expect(ResponseConversionHelper.toContentParts({ type: 'item_reference', id: 'ir1' } as never)).toEqual([]);
  });

  it('drops mcp_approval_request and mcp_approval_response', () => {
    // toContentParts filters out null parts, so both return empty arrays
    expect(ResponseConversionHelper.toContentParts({ type: 'mcp_approval_request', id: 'mar1' } as never)).toEqual([]);
    expect(ResponseConversionHelper.toContentParts({ type: 'mcp_approval_response', id: 'mar2' } as never)).toEqual([]);
  });

  // ── annotations / citations round-trip ───────────────────────────────────

  it('round-trips output_text annotations as citations', () => {
    const item: ResponseOutputItem = {
      type: 'message',
      id: 'msg-ann',
      role: 'assistant',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'see the docs',
          annotations: [
            { type: 'url_citation', url: 'https://docs.example.com', title: 'Docs', start_index: 0, end_index: 3 },
          ],
        },
      ],
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'text',
      text: 'see the docs',
      citations: [
        { type: 'web_location', url: 'https://docs.example.com', title: 'Docs', startIndex: 0, endIndex: 3 },
      ],
    });

    // Session → Response preserves url_citation
    const items = ResponseConversionHelper.toResponseItems('assistant', parts);
    const msg = items.find((i) => i && typeof i === 'object' && 'type' in i && i.type === 'message') as
      | { type: 'message'; content: Array<Record<string, unknown>> }
      | undefined;
    const outputText = (msg?.content[0] ?? {}) as Record<string, unknown>;
    expect(outputText.annotations).toMatchObject([
      { type: 'url_citation', url: 'https://docs.example.com' },
    ]);
  });

  it('round-trips file_citation annotations', () => {
    const item: ResponseOutputItem = {
      type: 'message',
      id: 'msg-fc',
      role: 'assistant',
      status: 'completed',
      content: [
        {
          type: 'output_text',
          text: 'file content',
          annotations: [
            { type: 'file_citation', file_id: 'f1', filename: 'test.ts', index: 0 },
          ],
        },
      ],
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    const textPart = parts.find((part) => part && typeof part === 'object' && 'type' in part && part.type === 'text') as
      | { type: 'text'; text: string; citations?: Array<Record<string, unknown>> }
      | undefined;
    expect(textPart?.citations).toMatchObject([
      { type: 'file_location', fileId: 'f1', fileName: 'test.ts' },
    ]);
  });

  // ── tool result with structured output ───────────────────────────────────

  it('converts function_call_output with structured content array', () => {
    const item: ResponseInputItem.FunctionCallOutput = {
      type: 'function_call_output',
      call_id: 'call-str',
      output: [
        { type: 'input_text', text: 'result part 1' },
        { type: 'input_text', text: 'result part 2' },
      ],
      status: 'completed',
    };

    const parts = ResponseConversionHelper.toContentParts(item);
    expect(parts[0]).toMatchObject({
      type: 'tool_result',
      id: 'call-str',
    });
    // The output array is JSON.stringified
    const result = JSON.parse((parts[0] as { result: string }).result);
    expect(result).toMatchObject([
      { type: 'text', text: 'result part 1' },
      { type: 'text', text: 'result part 2' },
    ]);
  });

  // ── input item conversions ───────────────────────────────────────────────

  it('builds input items for user role with mixed content', () => {
    const items = ResponseConversionHelper.toResponseInputItems({
      messageId: 'in-1',
      type: 'session_message',
      role: 'user',
      content: [
        { type: 'text', text: 'look at this' },
        {
          type: 'image',
          image: { sourceType: 'url', url: 'https://example.com/img.png' },
        },
      ],
    });

    expect(items).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          { type: 'input_text', text: 'look at this' },
          { type: 'input_image', detail: 'auto', image_url: 'https://example.com/img.png' },
        ],
      },
    ]);
  });

  it('converts tool result with providerSpecific RESPONSE marker', () => {
    const sessionMsg = {
      messageId: 'tr-ps',
      type: 'session_message' as const,
      role: 'tool' as const,
      content: [
        {
          type: 'tool_result' as const,
          id: 'comp-1',
          name: 'computer_call_output',
          result: JSON.stringify({ output: { type: 'input_text', text: 'screen' }, status: 'completed' }),
          providerSpecific: ApiModes.RESPONSE,
        },
      ],
    } satisfies Parameters<typeof ResponseConversionHelper.toResponseInputItems>[0];

    const items = ResponseConversionHelper.toResponseInputItems(sessionMsg);
    expect(items[0]).toMatchObject({
      type: 'computer_call_output',
      call_id: 'comp-1',
      status: 'completed',
    });
  });

  // ── output item grouping ─────────────────────────────────────────────────

  it('groups consecutive output_text + refusal into a single message', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'grp-1',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'text', text: 'hello' },
        { type: 'refusal', reason: 'cannot answer' },
      ],
    });

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      type: 'message',
      role: 'assistant',
      status: 'completed',
      content: [
        { type: 'output_text', text: 'hello', annotations: [] },
        { type: 'refusal', refusal: 'cannot answer' },
      ],
    });
  });

  it('separates tool_call from text into different items', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'sep-1',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'tool_call', id: 'tc-1', name: 'search', arguments: '{"q":"test"}' },
        { type: 'text', text: 'done' },
      ],
    });

    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({ type: 'function_call', call_id: 'tc-1' });
    expect(items[1]).toMatchObject({
      type: 'message',
      content: [{ type: 'output_text', text: 'done' }],
    });
  });

  // ── fallback for non-Response server_tool_use ────────────────────────────

  it('falls back to text for non-Response server_tool_use (assistant)', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'stu-fb-r',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'server_tool_use',
          name: 'anthropic_tool',
          data: '{"key":"val"}',
          providerSpecific: ApiModes.ANTHROPIC,
        },
      ],
    });

    // Fallback is wrapped inside a message envelope by toResponseOutputItems
    expect(items[0]).toMatchObject({
      type: 'message',
      role: 'assistant',
      content: [
        {
          type: 'output_text',
          text: 'Tool use: anthropic_tool with data {"key":"val"}',
        },
      ],
    });
  });

  it('falls back to text for non-Response server_tool_use (user/tool)', () => {
    const items = ResponseConversionHelper.toResponseInputItems({
      messageId: 'stu-fb-i',
      type: 'session_message',
      role: 'tool',
      content: [
        {
          type: 'server_tool_use',
          name: 'anthropic_tool',
          data: '{"key":"val"}',
          providerSpecific: ApiModes.ANTHROPIC,
        },
      ],
    });

    // Fallback is wrapped inside a message envelope by toResponseInputItems
    expect(items[0]).toMatchObject({
      type: 'message',
      role: 'user',
      content: [
        {
          type: 'input_text',
          text: 'Tool use: anthropic_tool with data {"key":"val"}',
        },
      ],
    });
  });
  // ── media fallback to text (assistant) ──────────────────────────────────

  it('demotes document to output_text in assistant', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'doc-fb',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'document',
          document: { sourceType: 'file_id', fileId: 'f-doc', fileName: 'doc.pdf' },
        },
      ],
    });

    const msg = items.find((i) => i.type === 'message') as { content: Array<Record<string, unknown>> } | undefined;
    expect(msg).toBeTruthy();
    const outputText = msg!.content[0];
    expect(outputText.type).toBe('output_text');
    expect(outputText.text).toContain('"fileId":"f-doc"');
  });

  it('demotes audio to output_text in assistant', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'audio-fb',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'audio', audio: { sourceType: 'base64', mimeType: 'audio/wav', data: 'AAAA' } },
      ],
    });

    const msg = items.find((i) => i.type === 'message') as { content: Array<Record<string, unknown>> } | undefined;
    expect(msg).toBeTruthy();
    expect(msg!.content[0]).toMatchObject({ type: 'output_text' });
    expect(msg!.content[0].text).toContain('"data":"AAAA"');
  });

  it('demotes video to output_text in assistant', () => {
    const items = ResponseConversionHelper.toResponseOutputItems({
      messageId: 'video-fb',
      type: 'session_message',
      role: 'assistant',
      content: [
        { type: 'video', video: { sourceType: 'url', url: 'https://example.com/v.mp4' } },
      ],
    });

    const msg = items.find((i) => i.type === 'message') as { content: Array<Record<string, unknown>> } | undefined;
    expect(msg).toBeTruthy();
    expect(msg!.content[0]).toMatchObject({ type: 'output_text' });
    expect(msg!.content[0].text).toContain('"url":"https://example.com/v.mp4"');
  });

  // ── media fallback to text (user/tool input) ────────────────────────────

  it('demotes audio to input_text in user input', () => {
    const items = ResponseConversionHelper.toResponseInputItems({
      messageId: 'audio-in',
      type: 'session_message',
      role: 'user',
      content: [
        { type: 'audio', audio: { sourceType: 'base64', mimeType: 'audio/mp3', data: 'bbbb' } },
      ],
    });

    const msg = items.find((i) => i.type === 'message') as { content: Array<Record<string, unknown>> } | undefined;
    expect(msg).toBeTruthy();
    expect(msg!.content[0]).toMatchObject({ type: 'input_text' });
    expect(msg!.content[0].text).toContain('"mimeType":"audio/mp3"');
  });

  it('demotes video to input_text in user input', () => {
    const items = ResponseConversionHelper.toResponseInputItems({
      messageId: 'video-in',
      type: 'session_message',
      role: 'user',
      content: [
        { type: 'video', video: { sourceType: 'url', url: 'https://example.com/v.mp4' } },
      ],
    });

    const msg = items.find((i) => i.type === 'message') as { content: Array<Record<string, unknown>> } | undefined;
    expect(msg).toBeTruthy();
    expect(msg!.content[0]).toMatchObject({ type: 'input_text' });
    expect(msg!.content[0].text).toContain('"url":"https://example.com/v.mp4"');
  });});
