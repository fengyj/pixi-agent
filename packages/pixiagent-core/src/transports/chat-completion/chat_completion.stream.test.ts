import { describe, expect, it, vi } from 'vitest';
import type { ChatCompletionChunk } from 'openai/resources/chat/completions';
import { ChatCompletionStreamProcessor } from './chat_completion.stream';

// ---------------------------------------------------------------------------
// Stream helper — inline generator to avoid type coercion ambiguity
// ---------------------------------------------------------------------------

function stream(...chunks: ChatCompletionChunk[]): AsyncIterable<ChatCompletionChunk> {
  return {
    [Symbol.asyncIterator]: () => {
      let i = 0;
      return {
        next: async () => {
          if (i >= chunks.length) return { done: true as const, value: undefined };
          return { done: false as const, value: chunks[i++] };
        },
      };
    },
  };
}

function baseChunk(overrides: Partial<ChatCompletionChunk> = {}): ChatCompletionChunk {
  return {
    id: 'chunk-1',
    model: 'gpt-4o-mini',
    created: 1,
    object: 'chat.completion.chunk',
    choices: [{ index: 0, finish_reason: null, delta: { content: '' }, logprobs: null }],
    ...overrides,
  } as ChatCompletionChunk;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ChatCompletionStreamProcessor', () => {
  // ── text accumulation ─────────────────────────────────────────────────

  it('accumulates streamed text into a final response', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { role: 'assistant', content: 'hi ' }, logprobs: null }] }),
        baseChunk({ choices: [{ index: 0, finish_reason: 'stop', delta: { content: 'there' }, logprobs: null }] }),
      ),
    );
    expect(response.choices[0]?.message.content).toBe('hi there');
    expect(response.choices[0]?.finish_reason).toBe('stop');
  });

  it('waits for chunk callbacks to finish before resolving', async () => {
    const processor = new ChatCompletionStreamProcessor();
    let release!: () => void;
    const gate = new Promise<void>((r) => { release = r; });
    let settled = false;

    const result = processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { role: 'assistant', content: 'hi' }, logprobs: null }] }),
        baseChunk({ choices: [{ index: 0, finish_reason: 'stop', delta: { content: 'there' }, logprobs: null }] }),
      ),
      { onChunk: async () => { await gate; }, onFinish: async () => {} },
    );

    result.then(() => { settled = true; });
    await Promise.resolve();
    expect(settled).toBe(false);
    release();
    await result;
    expect(settled).toBe(true);
  });

  // ── edge cases ────────────────────────────────────────────────────────

  it('skips chunks with empty choices array', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(
      stream(
        baseChunk({ choices: [] }),
        baseChunk({ choices: [{ index: 0, finish_reason: 'stop', delta: { content: 'hello' }, logprobs: null }] }),
      ),
    );
    expect(response.choices[0]?.message.content).toBe('hello');
  });

  it('extracts usage from chunk', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(
      stream(
        baseChunk({ usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 } } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
      ),
    );
    expect(response.usage).toEqual({ prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 });
  });

  it('updates finish_reason from chunk', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(
      stream(baseChunk({ choices: [{ index: 0, finish_reason: 'length', delta: { content: 'x' }, logprobs: null }] })),
    );
    expect(response.choices[0]?.finish_reason).toBe('length');
  });

  // ── refusal delta ─────────────────────────────────────────────────────

  it('accumulates refusal delta', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { refusal: 'I cannot' }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { refusal: ' comply' }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: 'stop', delta: {}, logprobs: null }] }),
      ),
    );
    expect(response.choices[0]?.message.refusal).toBe('I cannot comply');
  });

  it('leaves refusal undefined when none present', async () => {
    const processor = new ChatCompletionStreamProcessor();
    const response = await processor.process(stream(baseChunk()));
    expect(response.choices[0]?.message.refusal).toBeUndefined();
  });

  // ── tool_call accumulation ────────────────────────────────────────────

  it('accumulates a single tool call across multiple chunks', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'search', arguments: '' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, function: { arguments: '{"q":"pixi"}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
      ),
    );
    expect(response.choices[0]?.message.tool_calls).toHaveLength(1);
    expect(response.choices[0]?.message.tool_calls![0]).toMatchObject({
      id: 'call-1', type: 'function', function: { name: 'search', arguments: '{"q":"pixi"}' },
    });
  });

  it('accumulates multiple tool calls in order', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'search', arguments: '{"q":"a"}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 1, id: 'c2', function: { name: 'lookup', arguments: '{"id":"b"}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
      ),
    );
    expect(response.choices[0]?.message.tool_calls).toHaveLength(2);
  });

  it('includes text content alongside tool calls', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { role: 'assistant', content: 'Thinking' }, logprobs: null }] }),
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'search', arguments: '{"q":"test"}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: 'tool_calls', delta: {}, logprobs: null }] }),
      ),
    );
    expect(response.choices[0]?.message.content).toBe('Thinking');
    expect(response.choices[0]?.message.tool_calls).toHaveLength(1);
    expect(response.choices[0]?.finish_reason).toBe('tool_calls');
  });

  // ── onChunk callback ──────────────────────────────────────────────────

  it('emits onChunk for text deltas', async () => {
    const onChunk = vi.fn();
    const processor = new ChatCompletionStreamProcessor();

    await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { role: 'assistant', content: 'Hi' }, logprobs: null }] }),
        baseChunk({ choices: [{ index: 0, finish_reason: 'stop', delta: { content: ' there' }, logprobs: null }] }),
      ),
      { onChunk },
    );

    expect(onChunk).toHaveBeenCalledTimes(2);
    expect(onChunk).toHaveBeenNthCalledWith(1, expect.objectContaining({
      contentPartChunk: { type: 'text', text: 'Hi' },
    }));
  });

  it('emits onChunk for tool call deltas', async () => {
    const onChunk = vi.fn();
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');

    await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, id: 'call-1', function: { name: 'search', arguments: '{"q":"test"}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        baseChunk({ choices: [{ index: 0, finish_reason: 'tool_calls', delta: {}, logprobs: null }] }),
      ),
      { onChunk },
    );

    expect(onChunk).toHaveBeenCalledTimes(1);
    expect(onChunk).toHaveBeenCalledWith(expect.objectContaining({
      contentPartChunk: { type: 'tool_call', id: 'call-1', name: 'search', arguments: '{"q":"test"}' },
    }));
  });

  // ── error: tool call index mismatch ───────────────────────────────────

  it('throws on tool call index out of order (index 1 before 0)', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    await expect(
      processor.process(
        stream(
          baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 1, id: 'c2', function: { name: 'b', arguments: '{}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        ),
      ),
    ).rejects.toThrow('out of order or with gaps');
  });

  it('throws on tool call index gap (0 then 2)', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    await expect(
      processor.process(
        stream(
          baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0, id: 'c1', function: { name: 'a', arguments: '{}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
          baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 2, id: 'c3', function: { name: 'c', arguments: '{}' } }] }, logprobs: null }] } as Partial<ChatCompletionChunk> as ChatCompletionChunk),
        ),
      ),
    ).rejects.toThrow('out of order or with gaps');
  });

  it('handles tool call with empty id/name/arguments defaults', async () => {
    const processor = new ChatCompletionStreamProcessor(undefined, 'https://api.openai.com');
    const response = await processor.process(
      stream(
        baseChunk({ choices: [{ index: 0, finish_reason: null, delta: { tool_calls: [{ index: 0 }] }, logprobs: null }] } as unknown as ChatCompletionChunk),
      ),
    );
    expect(response.choices[0]?.message.tool_calls![0]).toMatchObject({ id: '', function: { name: '', arguments: '' } });
  });
});
