import { describe, expect, it } from 'vitest';
import type { ChatCompletionChunk } from 'openai/resources/chat/completions';
import { ChatCompletionStreamProcessor } from './chat_completion.stream';

async function* makeStream(): AsyncIterable<ChatCompletionChunk> {
  yield {
    id: 'chunk-1',
    model: 'gpt-4o-mini',
    created: 1,
    object: 'chat.completion.chunk',
    choices: [
      {
        index: 0,
        finish_reason: null,
        delta: { role: 'assistant', content: 'hi ' },
        logprobs: null,
      },
    ],
  } as ChatCompletionChunk;

  yield {
    id: 'chunk-1',
    model: 'gpt-4o-mini',
    created: 1,
    object: 'chat.completion.chunk',
    choices: [
      {
        index: 0,
        finish_reason: 'stop',
        delta: { content: 'there' },
        logprobs: null,
      },
    ],
  } as ChatCompletionChunk;
}

describe('ChatCompletionStreamProcessor', () => {
  it('accumulates streamed text into a final chat completion response', async () => {
    const processor = new ChatCompletionStreamProcessor();

    const response = await processor.process(makeStream());

    expect(response.id).toBe('chunk-1');
    expect(response.choices[0]?.message.content).toEqual('hi there');
    expect(response.choices[0]?.finish_reason).toBe('stop');
  });

  it('waits for chunk callbacks to finish before resolving', async () => {
    const processor = new ChatCompletionStreamProcessor();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });

    let settled = false;
    const result = processor.process(makeStream(), {
      onChunk: async () => {
        await gate;
      },
      onFinish: async () => {},
    });

    result.then(() => {
      settled = true;
    });

    await Promise.resolve();
    expect(settled).toBe(false);

    release();
    await result;
    expect(settled).toBe(true);
  });
});
