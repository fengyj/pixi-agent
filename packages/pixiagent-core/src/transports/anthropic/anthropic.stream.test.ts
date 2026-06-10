import { describe, expect, it, vi } from 'vitest';
import type {
  Message,
  MessageStreamEvent,
  ContentBlock,
} from '@anthropic-ai/sdk/resources/messages/messages';
import { StreamDataExtractor } from '../base';
import { AnthropicStreamProcessor } from './anthropic.stream';

// ---------------------------------------------------------------------------
// Factory helpers — each call returns a fresh copy to avoid cross-test mutation
// ---------------------------------------------------------------------------

function msgStart(opts?: Partial<MessageStreamEvent & { message: Partial<Message> }>): MessageStreamEvent {
  return {
    type: 'message_start',
    message: {
      id: 'msg_01',
      type: 'message',
      role: 'assistant',
      model: 'claude-sonnet-4-20250514',
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 10, output_tokens: 0 },
      ...(opts?.message ?? {}),
    },
    ...(opts ?? {}),
  } as MessageStreamEvent;
}

function msgDelta(opts?: { stop_reason?: string | null; usage?: Partial<Message['usage']> }): MessageStreamEvent {
  return {
    type: 'message_delta',
    delta: { stop_reason: opts?.stop_reason ?? 'end_turn', stop_sequence: null },
    usage: { input_tokens: 10, output_tokens: 25, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, ...(opts?.usage ?? {}) },
  } as MessageStreamEvent;
}

const msgStop: MessageStreamEvent = { type: 'message_stop' };

async function* events(evts: MessageStreamEvent[]): AsyncIterable<MessageStreamEvent> {
  for (const e of evts) {
    yield e;
  }
}

function emptyExtractor(): StreamDataExtractor<Message> {
  return new StreamDataExtractor(
    {
      content: [] as ContentBlock[],
      id: '',
      container: null,
      model: '',
      role: 'assistant',
      stop_details: null,
      stop_reason: null,
      stop_sequence: null,
      type: 'message',
      usage: { input_tokens: 0, output_tokens: 0 },
    } as Message,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('AnthropicStreamProcessor', () => {
  // ────── process() – complete flows ────────────────────────────────────

  describe('process() - complete flows', () => {
    it('accumulates a simple text response', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'Hello' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ' World' } },
          { type: 'content_block_stop', index: 0 },
          msgDelta(),
          msgStop,
        ]),
      );

      expect(result.content.content).toHaveLength(1);
      expect(result.content.content[0]).toMatchObject({ type: 'text', text: 'Hello World' });
    });

    it('accumulates text with tool_use and parses input JSON', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'Here:' } },
          { type: 'content_block_stop', index: 0 },
          { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'tu_01', name: 'search', input: '' } },
          { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"q":"pixi' } },
          { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '"}' } },
          { type: 'content_block_stop', index: 1 },
          msgDelta(),
          msgStop,
        ]),
      );

      expect(result.content.content).toHaveLength(2);
      expect(result.content.content[1]).toMatchObject({ type: 'tool_use', id: 'tu_01', name: 'search', input: { q: 'pixi' } });
    });

    it('accumulates thinking blocks with signature', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: 'Let me think', signature: 'sig_1' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: ' step' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'signature_delta', signature: 'sig_2' } },
          { type: 'content_block_stop', index: 0 },
          { type: 'content_block_start', index: 1, content_block: { type: 'text', text: 'Final answer' } },
          { type: 'content_block_stop', index: 1 },
          msgDelta(),
          msgStop,
        ]),
      );

      // Note: signature_delta appends to the existing signature
      expect(result.content.content[0]).toMatchObject({ type: 'thinking', thinking: 'Let me think step', signature: 'sig_1sig_2' });
      expect(result.content.content[1]).toMatchObject({ type: 'text', text: 'Final answer' });
    });

    it('handles citations_delta', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const citation = { type: 'web_search_result_location' as const, cited_text: 'Pixi', title: 'Pixi Docs', url: 'https://example.com' };
      const result = await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'Cited' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'citations_delta', citation } },
          { type: 'content_block_stop', index: 0 },
          msgDelta(),
          msgStop,
        ]),
      );

      const block = result.content.content[0] as ContentBlock & { type: 'text'; citations: unknown[] };
      expect(block.citations).toHaveLength(1);
      expect(block.citations![0]).toMatchObject(citation);
    });

    it('maps stop_reason end_turn → STOP', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([msgStart(), msgDelta({ stop_reason: 'end_turn' }), msgStop]));
      expect(result.modelResponseInfo?.stopReason).toBe('stop');
    });

    it('maps stop_reason tool_use → TOOL_CALL', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([msgStart(), msgDelta({ stop_reason: 'tool_use' }), msgStop]));
      expect(result.modelResponseInfo?.stopReason).toBe('tool_call');
    });

    it('maps stop_reason max_tokens → MAX_TOKENS', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([msgStart(), msgDelta({ stop_reason: 'max_tokens' }), msgStop]));
      expect(result.modelResponseInfo?.stopReason).toBe('max_tokens');
    });

    it('maps stop_reason stop_sequence → STOP', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([msgStart(), msgDelta({ stop_reason: 'stop_sequence' }), msgStop]));
      expect(result.modelResponseInfo?.stopReason).toBe('stop');
    });

    it('extracts usage from message_delta', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([
        msgStart(),
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { input_tokens: 15, output_tokens: 30, cache_creation_input_tokens: 5, cache_read_input_tokens: 3 } } as MessageStreamEvent,
        msgStop,
      ]));
      expect(result.modelResponseInfo?.usage).toMatchObject({ inputTokens: 15, outputTokens: 30, cacheCreatedTokens: 5, cacheReadTokens: 3 });
    });

    it('calls onChunk callback for each text delta', async () => {
      const onChunk = vi.fn();
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');

      await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'Hi' } },
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: ' there' } },
          { type: 'content_block_stop', index: 0 },
          msgDelta(),
          msgStop,
        ]),
        { onChunk },
      );

      expect(onChunk).toHaveBeenCalledTimes(2);
      expect(onChunk).toHaveBeenNthCalledWith(1, expect.objectContaining({ contentPartIndex: 0, contentPartChunk: { type: 'text', text: 'Hi' }, chunkIndex: 0 }));
      expect(onChunk).toHaveBeenNthCalledWith(2, expect.objectContaining({ contentPartIndex: 0, contentPartChunk: { type: 'text', text: ' there' }, chunkIndex: 1 }));
    });

    it('converts server_tool_use start to a content part', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(
        events([
          msgStart(),
          { type: 'content_block_start', index: 0, content_block: { type: 'server_tool_use', id: 'srv_01', name: 'custom-tool', input: { test: true } } } as MessageStreamEvent,
          { type: 'content_block_stop', index: 0 },
          msgDelta(),
          msgStop,
        ]),
      );
      expect(result.content.content[0]).toMatchObject({ type: 'server_tool_use', name: 'custom-tool' });
    });
  });

  // ────── message_start ─────────────────────────────────────────────────

  describe('message_start', () => {
    it('sets all fields from message_start event', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      await processor.handleEvent(msgStart(), extractor);
      expect(extractor.accumulatedData).toMatchObject({ id: 'msg_01', model: 'claude-sonnet-4-20250514', role: 'assistant' });
    });
  });

  // ────── message_delta edge cases ──────────────────────────────────────

  describe('message_delta edge cases', () => {
    it('updates container / stop_details / stop_sequence when present', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      await processor.handleEvent(
        {
          type: 'message_delta',
          delta: { container: { key: 'cont-1' }, stop_reason: 'end_turn', stop_details: { explanation: 'done' }, stop_sequence: 'seq-123' },
          usage: { input_tokens: 5, output_tokens: 10, cache_creation_input_tokens: 1, cache_read_input_tokens: 2, server_tool_use: { tool_use: { id: 'srv', name: 'tool', input: {} } } },
        } as unknown as MessageStreamEvent,
        extractor,
      );
      expect(extractor.accumulatedData).toMatchObject({ container: { key: 'cont-1' }, stop_details: { explanation: 'done' }, stop_sequence: 'seq-123' });
      expect(extractor.accumulatedData.usage.server_tool_use).toEqual({ tool_use: { id: 'srv', name: 'tool', input: {} } });
    });

    it('skips update when delta container/stop_details are falsy', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).container = { key: 'orig' };
      (extractor.accumulatedData as any).stop_details = { explanation: 'orig' };

      await processor.handleEvent(
        { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { input_tokens: 0, output_tokens: 0 } } as unknown as MessageStreamEvent,
        extractor,
      );
      expect(extractor.accumulatedData.container).toEqual({ key: 'orig' });
      expect(extractor.accumulatedData.stop_details).toEqual({ explanation: 'orig' });
    });
  });

  // ────── handleEvent() – error paths ───────────────────────────────────

  describe('handleEvent() - error paths', () => {
    it('throws on content_block_start with mismatched index', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'text', text: 'first' });

      await expect(
        processor.handleEvent(
          { type: 'content_block_start', index: 0, content_block: { type: 'text', text: 'oops' } } as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('does not match expected index');
    });

    it('throws on content_block_delta when block index exceeds content length (no prior start)', async () => {
      // For a new block (no prior content_block_start), the delta handler's merge callback
      // detects the mismatch. First delta creates the block; second delta triggers merge validation.
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      // Add a real block at index 0 so that index 5 is in-bounds for content length check
      (extractor.accumulatedData as any).content.push({ type: 'text', text: 'existing' });

      // First delta for index 1 — key is new (content.length=1, event.index=1, error in merge checks 1 <= 1 → true)
      await processor.handleEvent(
        { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'x' } } as unknown as MessageStreamEvent,
        extractor,
      );

      // Second delta for index 1 — now merge runs, content.length=1 <= event.index=1 → throws
      await expect(
        processor.handleEvent(
          { type: 'content_block_delta', index: 1, delta: { type: 'text_delta', text: 'y' } } as unknown as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('exceeds current content length');
    });

    it('throws on text_delta for a non-text block (tool_use)', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'tool_use', id: 'tu_01', name: 'search', input: {} });

      await processor.handleEvent(
        { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'oops' } } as unknown as MessageStreamEvent,
        extractor,
      );
      // Second delta for same index triggers merge which validates the type
      await expect(
        processor.handleEvent(
          { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'oops2' } } as unknown as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('which is not of type \'text\'');
    });

    it('throws on thinking_delta for a non-thinking block (text)', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'text', text: 'hello' });

      await processor.handleEvent(
        { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'hmm' } } as unknown as MessageStreamEvent,
        extractor,
      );
      await expect(
        processor.handleEvent(
          { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'hmm2' } } as unknown as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('which is not of type \'thinking\'');
    });

    it('throws on input_json_delta for a non-tool_use block (text)', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'text', text: 'hello' });

      await processor.handleEvent(
        { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{}' } } as unknown as MessageStreamEvent,
        extractor,
      );
      await expect(
        processor.handleEvent(
          { type: 'content_block_delta', index: 0, delta: { type: 'input_json_delta', partial_json: '{}' } } as unknown as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('which is not of type \'tool_use\'');
    });

    it('throws on citations_delta for a non-text block (tool_use)', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'tool_use', id: 'tu_01', name: 'search', input: {} });

      await processor.handleEvent(
        { type: 'content_block_delta', index: 0, delta: { type: 'citations_delta', citation: { type: 'web_search_result_location', cited_text: 'x', title: 'y', url: 'z' } } } as unknown as MessageStreamEvent,
        extractor,
      );
      await expect(
        processor.handleEvent(
          { type: 'content_block_delta', index: 0, delta: { type: 'citations_delta', citation: { type: 'web_search_result_location', cited_text: 'x', title: 'y', url: 'z' } } } as unknown as MessageStreamEvent,
          extractor,
        ),
      ).rejects.toThrow('which is not of type \'text\'');
    });

    it('throws on content_block_stop with out-of-range index', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      await expect(
        processor.handleEvent({ type: 'content_block_stop', index: 99 } as MessageStreamEvent, extractor),
      ).rejects.toThrow('exceeds current content length');
    });

    it('throws on content_block_stop when tool_use input JSON is invalid', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const extractor = emptyExtractor();
      (extractor.accumulatedData as any).content.push({ type: 'tool_use', id: 'tu_01', name: 'search', input: 'not-json{' });
      await expect(
        processor.handleEvent({ type: 'content_block_stop', index: 0 } as MessageStreamEvent, extractor),
      ).rejects.toThrow('Failed to parse input JSON');
    });

    it('returns a result even without message_stop event', async () => {
      const processor = new AnthropicStreamProcessor(undefined, 'https://api.anthropic.com');
      const result = await processor.process(events([msgStart()]));
      expect(result.content.content).toHaveLength(0);
    });
  });
});
