import { describe, expect, it } from 'vitest';
import { ResponseTransport, ConvertHelper } from './response';
import type {
  ResponseApiMessage,
  SessionMessage,
} from '../message';
import type {
  ResponseInputItem,
  ResponseOutputItem,
} from 'openai/resources/responses/responses';

describe('ResponseTransport conversion', () => {
  const transport = new ResponseTransport(undefined, 'test-api-key');

  it('converts response raw message output into session parts', () => {
    const raw: ResponseApiMessage = {
      messageId: 'resp-1',
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'message',
          id: 'resp-1',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'hello response', annotations: [] }],
        },
        {
          type: 'function_call',
          id: 'func-1',
          call_id: 'call-1',
          name: 'search',
          arguments: '{"q":"pixi"}',
          status: 'completed',
        },
      ],
    };

    const session = transport.convertFromRawMessage(raw);

    expect(session.messageId).toBe('resp-1');
    expect(session.type).toBe('session_message');
    expect(session.role).toBe('assistant');
    expect(Array.isArray(session.content)).toBe(true);
    expect(session.content).toMatchObject([
      { type: 'text', text: 'hello response' },
      {
        type: 'tool_call',
        id: 'call-1',
        name: 'search',
        arguments: '{"q":"pixi"}',
      },
    ]);
  });

  it('converts assistant tool_call and text session content into response raw message items', () => {
    const sessionMessage: SessionMessage = {
      messageId: 'resp-2',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: 'call-2',
          name: 'run',
          arguments: '{"command":"date"}',
        },
        { type: 'text', text: 'done' },
      ],
    };

    const raw = transport.convertToRawMessage(sessionMessage);

    expect(raw).toEqual({
      messageId: 'resp-2',
      type: 'response_api_message',
      role: 'assistant',
      metadata: undefined,
      content: [
        {
          type: 'function_call',
          id: 'call-2',
          call_id: 'call-2',
          name: 'run',
          arguments: '{"command":"date"}',
          status: 'completed',
        },
        {
          type: 'message',
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: 'done', annotations: [] }],
        },
      ],
    });
  });

  it('converts response raw function_call_output into tool_result session content', () => {
    const raw: ResponseApiMessage = {
      messageId: 'resp-3',
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'function_call_output',
          call_id: 'call-3',
          output: 'completed successfully',
          status: 'completed',
        },
      ],
    };

    const session = transport.convertFromRawMessage(raw);

    expect(session).toEqual({
      messageId: 'resp-3',
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_result',
          id: 'call-3',
          result: 'completed successfully',
        },
      ],
      metadata: undefined,
    });
  });

  it('converts tool session message with tool_result and user content into response raw items', () => {
    const sessionMessage: SessionMessage = {
      messageId: 'resp-4',
      type: 'session_message',
      role: 'tool',
      content: [
        {
          type: 'tool_result',
          id: 'call-4',
          result: 'tool output',
        },
        { type: 'text', text: 'fallback text' },
      ],
    };

    const raw = transport.convertToRawMessage(sessionMessage);

    expect(raw).toEqual({
      messageId: 'resp-4',
      type: 'response_api_message',
      role: 'tool',
      metadata: undefined,
      content: [
        {
          type: 'function_call_output',
          call_id: 'call-4',
          output: 'tool output',
          status: 'completed',
        },
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'fallback text' }],
        },
      ],
    });
  });
});

describe('ConvertHelper conversion', () => {
  it('maps a response input message to content parts', () => {
    const item: ResponseInputItem.Message = {
      type: 'message',
      role: 'user',
      content: [{ type: 'input_text', text: 'hello world' }],
    };

    expect(ConvertHelper.toContentParts(item)).toEqual([
      {
        type: 'text',
        text: 'hello world',
      },
    ]);
  });

  it('maps a response function_call to a tool_call content part', () => {
    const item: ResponseOutputItem = {
      type: 'function_call',
      id: 'func-1',
      call_id: 'call-1',
      name: 'search',
      arguments: '{"q":"pixi"}',
      status: 'completed',
    };

    expect(ConvertHelper.toContentParts(item)).toEqual([
      {
        type: 'tool_call',
        id: 'call-1',
        name: 'search',
        arguments: '{"q":"pixi"}',
      },
    ]);
  });

  it('builds response items from content parts for a user role', () => {
    const items = ConvertHelper.toResponseItems('user', [
      { type: 'text', text: 'user message' },
      {
        type: 'image',
        image: {
          sourceType: 'url',
          url: 'https://example.com/image.png',
        },
      },
    ]);

    expect(items).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          { type: 'input_text', text: 'user message' },
          {
            type: 'input_image',
            detail: 'auto',
            image_url: 'https://example.com/image.png',
          },
        ],
      },
    ]);
  });

  it('builds response items from content parts for an assistant role with a tool call', () => {
    const items = ConvertHelper.toResponseItems('assistant', [
      {
        type: 'tool_call',
        id: 'tool-1',
        name: 'run',
        arguments: '{"cmd":"ls"}',
      },
      { type: 'text', text: 'ok' },
    ]);

    expect(items).toEqual([
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
