import { describe, expect, it } from 'vitest';
import { ResponseConversionHelper } from './response-conversion';
import type { ResponseInputItem, ResponseOutputItem } from 'openai/resources/responses/responses';

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
});
