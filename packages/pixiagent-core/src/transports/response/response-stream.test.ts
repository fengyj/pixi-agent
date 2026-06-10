import { describe, expect, it, vi } from 'vitest';
import { ResponseStreamProcessor } from './response-stream';
import { StreamDataExtractor } from '../base';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function completedResponse(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'resp-1',
    object: 'response',
    created_at: 0,
    status: 'completed',
    error: null,
    incomplete_details: null,
    model: 'gpt-4o-mini',
    output: [],
    parallel_tool_calls: true,
    temperature: 1,
    tool_choice: 'auto',
    tools: [],
    top_p: 1,
    max_output_tokens: null,
    previous_response_id: null,
    reasoning: null,
    text: { format: null },
    truncation: 'disabled',
    usage: null,
    ...overrides,
  };
}

async function* streamFrom(...events: Record<string, unknown>[]) {
  for (const e of events) {
    yield e as never;
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ResponseStreamProcessor', () => {
  describe('process() - public API', () => {
    it('processes function_call with arguments delta', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const response = await processor.process(
        streamFrom(
          { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'call-1', call_id: 'call-1', name: 'search', arguments: '', status: 'in_progress' } },
          { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"q":"pixi"}' },
          { type: 'response.completed', response: completedResponse({ output: [{ type: 'function_call', id: 'call-1', call_id: 'call-1', name: 'search', arguments: '{"q":"pixi"}', status: 'completed' }] }) },
        ),
      );
      expect(response).toMatchObject({ id: 'resp-1', output: [expect.objectContaining({ type: 'function_call', name: 'search', arguments: '{"q":"pixi"}' })] });
    });

    it('processes message output with text deltas', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const response = await processor.process(
        streamFrom(
          { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } },
          { type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } },
          { type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'Hello ' },
          { type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'World!' },
          { type: 'response.completed', response: completedResponse({ output: [{ type: 'message', id: 'msg-1', role: 'assistant', status: 'completed', content: [{ type: 'output_text', text: 'Hello World!', annotations: [] }] }] }) },
        ),
      );
      expect(response.output[0]).toMatchObject({ type: 'message', content: [expect.objectContaining({ type: 'output_text', text: 'Hello World!' })] });
    });

    it('handles response.incomplete and response.failed as terminal events', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const incomplete = await processor.process(streamFrom({ type: 'response.incomplete', response: completedResponse({ status: 'incomplete' }) }));
      expect(incomplete.status).toBe('incomplete');
      const failed = await processor.process(streamFrom({ type: 'response.failed', response: completedResponse({ status: 'failed' }) }));
      expect(failed.status).toBe('failed');
    });

    it('throws if stream ends without terminal response event', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      await expect(processor.process(streamFrom({ type: 'response.created' }))).rejects.toThrow('ended without a terminal response event');
    });

    it('skips lifecycle events (created, queued, in_progress)', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const response = await processor.process(streamFrom({ type: 'response.created' }, { type: 'response.queued' }, { type: 'response.in_progress' }, { type: 'response.completed', response: completedResponse() }));
      expect(response.status).toBe('completed');
    });

    it('emits onChunk for output_text deltas', async () => {
      const onChunk = vi.fn();
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');

      await processor.process(
        streamFrom(
          { type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } },
          { type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } },
          { type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'Hi' },
          { type: 'response.completed', response: completedResponse() },
        ),
        { onChunk },
      );

      expect(onChunk).toHaveBeenCalledTimes(1);
      expect(onChunk).toHaveBeenCalledWith(expect.objectContaining({ contentPartChunk: { type: 'text', text: 'Hi' } }));
    });
  });

  // -----------------------------------------------------------------------
  // handleEvent() – detailed event handling
  // -----------------------------------------------------------------------

  describe('handleEvent() - text output', () => {
    it('accumulates output text deltas across multiple chunks', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } } as never, extractor);
      await processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } } as never, extractor);
      await processor.handleEvent({ type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: 'Hello' } as never, extractor);
      await processor.handleEvent({ type: 'response.output_text.delta', output_index: 0, content_index: 0, delta: ' World' } as never, extractor);

      const msg = extractor.accumulatedData.content[0] as { type: string; content: Array<{ type: string; text: string }> };
      expect(msg.content[0].text).toBe('Hello World');
    });
  });

  describe('handleEvent() - refusal delta', () => {
    it('accumulates refusal delta', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } } as never, extractor);
      await processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'refusal', refusal: '' } } as never, extractor);
      await processor.handleEvent({ type: 'response.refusal.delta', output_index: 0, content_index: 0, delta: 'I cannot ' } as never, extractor);
      await processor.handleEvent({ type: 'response.refusal.delta', output_index: 0, content_index: 0, delta: 'do that' } as never, extractor);

      const msg = extractor.accumulatedData.content[0] as { type: string; content: Array<{ type: string; refusal: string }> };
      expect(msg.content[0].refusal).toBe('I cannot do that');
    });
  });

  describe('handleEvent() - reasoning_text delta', () => {
    it('accumulates reasoning text deltas', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'reasoning', id: 'reason-1', status: 'in_progress', content: [] } } as never, extractor);
      await processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'reasoning_text', text: '' } } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_text.delta', output_index: 0, content_index: 0, delta: 'step ' } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_text.delta', output_index: 0, content_index: 0, delta: 'by step' } as never, extractor);

      const reasoning = extractor.accumulatedData.content[0] as { type: string; content: Array<{ type: string; text: string }> };
      expect(reasoning.content![0].text).toBe('step by step');
    });
  });

  describe('handleEvent() - reasoning summary', () => {
    it('accumulates reasoning summary part and text deltas', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'reasoning', id: 'reason-1', status: 'in_progress' } } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_summary_part.added', output_index: 0, summary_index: 0, part: { type: 'summary_text', text: '' } } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_summary_text.delta', output_index: 0, summary_index: 0, delta: 'Summary:' } as never, extractor);

      const reasoning = extractor.accumulatedData.content[0] as { type: string; summary: Array<{ type: string; text: string }> };
      expect(reasoning.summary).toHaveLength(1);
      expect(reasoning.summary![0].text).toBe('Summary:');
    });
  });

  describe('handleEvent() - output_text.annotation.added', () => {
    it('accumulates annotations on output text', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } } as never, extractor);
      await processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } } as never, extractor);
      await processor.handleEvent({ type: 'response.output_text.annotation.added', output_index: 0, content_index: 0, annotation: { type: 'url_citation', url_citation: { url: 'https://example.com', title: 'Example' } } } as never, extractor);

      const msg = extractor.accumulatedData.content[0] as { type: string; content: Array<{ type: string; annotations: unknown[] }> };
      expect(msg.content[0].annotations).toHaveLength(1);
      expect(msg.content[0].annotations![0]).toMatchObject({ type: 'url_citation' });
    });
  });

  describe('handleEvent() - output_item.done', () => {
    it('replaces item on output_item.done', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'call-1', call_id: 'call-1', name: 'search', arguments: '', status: 'in_progress' } } as never, extractor);
      await processor.handleEvent({ type: 'response.output_item.done', output_index: 0, item: { type: 'function_call', id: 'call-1', call_id: 'call-1', name: 'search', arguments: '{"q":"pixi"}', status: 'completed' } } as never, extractor);

      expect(extractor.accumulatedData.content[0]).toMatchObject({ type: 'function_call', arguments: '{"q":"pixi"}', status: 'completed' });
    });
  });

  describe('handleEvent() - content_part.added for reasoning item', () => {
    it('adds reasoning_text part to reasoning item', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'reasoning', id: 'r-1', status: 'in_progress' } } as never, extractor);
      await processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'reasoning_text', text: 'thinking...' } } as never, extractor);

      const item = extractor.accumulatedData.content[0] as { type: string; content: Array<{ type: string; text: string }> };
      expect(item.content).toHaveLength(1);
      expect(item.content![0]).toMatchObject({ type: 'reasoning_text', text: 'thinking...' });
    });
  });

  describe('handleEvent() - simple no-op events', () => {
    it('returns without action for done events', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);

      await processor.handleEvent({ type: 'response.content_part.done', output_index: 0, content_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.output_text.done', output_index: 0, content_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.refusal.done', output_index: 0, content_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_text.done', output_index: 0, content_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_summary_text.done', output_index: 0, summary_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.reasoning_summary_part.done', output_index: 0, summary_index: 0 } as never, extractor);
      await processor.handleEvent({ type: 'response.function_call_arguments.done', output_index: 0 } as never, extractor);

      expect(extractor.accumulatedData.content).toHaveLength(0);
    });
  });

  // -----------------------------------------------------------------------
  // Error events
  // -----------------------------------------------------------------------

  describe('handleEvent() - error events', () => {
    it('throws retriable error for rate_limit_exceeded', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await expect(processor.handleEvent({ type: 'error', code: 'rate_limit_exceeded', message: 'Too fast' } as never, extractor)).rejects.toThrow('Too fast');
    });

    it('throws retriable error for vector_store_timeout', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await expect(processor.handleEvent({ type: 'error', code: 'vector_store_timeout', message: 'Timeout' } as never, extractor)).rejects.toThrow('Timeout');
    });

    it('throws response error for server_error', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await expect(processor.handleEvent({ type: 'error', code: 'server_error', message: 'Internal error' } as never, extractor)).rejects.toThrow('Internal error');
    });

    it('throws invalid message error for unknown error code', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await expect(processor.handleEvent({ type: 'error', code: 'unknown_code', message: 'Unknown' } as never, extractor)).rejects.toThrow('Error event received from response stream');
    });
  });

  // -----------------------------------------------------------------------
  // Structural error paths
  // -----------------------------------------------------------------------

  describe('handleEvent() - structural error paths', () => {
    it('throws when output_item.added has mismatched index', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await expect(
        processor.handleEvent({ type: 'response.output_item.added', output_index: 5, item: { type: 'function_call', id: 'c-1', call_id: 'c-1', name: 'x', arguments: '' } } as never, extractor),
      ).rejects.toThrow('Received output item for non-existing index');
    });

    it('throws when content_part.added targets unsupported item type (function_call)', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'c-1', call_id: 'c-1', name: 'x', arguments: '' } } as never, extractor);
      await expect(processor.handleEvent({ type: 'response.content_part.added', output_index: 0, content_index: 0, part: { type: 'output_text', text: '' } } as never, extractor)).rejects.toThrow('Received content part for unsupported item type');
    });

    it('throws when reasoning_summary_part.added targets non-reasoning item', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, undefined);
      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'c-1', call_id: 'c-1', name: 'x', arguments: '' } } as never, extractor);
      await expect(processor.handleEvent({ type: 'response.reasoning_summary_part.added', output_index: 0, summary_index: 0, part: { type: 'summary_text', text: '' } } as never, extractor)).rejects.toThrow('Received reasoning summary part for a non-reasoning output item');
    });

    it('throws when function_call_arguments.delta targets non-function_call item', async () => {
      const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');
      const extractor = new StreamDataExtractor({ content: [] as never[], response: undefined as never }, { onChunk: vi.fn(), onFinish: vi.fn() as never });
      await processor.handleEvent({ type: 'response.output_item.added', output_index: 0, item: { type: 'message', id: 'msg-1', role: 'assistant', status: 'in_progress', content: [] } } as never, extractor);

      // The toContentPart callback validates the type and runs when a callback is provided
      await expect(
        processor.handleEvent({ type: 'response.function_call_arguments.delta', output_index: 0, delta: '{}' } as never, extractor),
      ).rejects.toThrow('Received function call arguments delta for non-existing or non-function call item');
    });
  });
});
