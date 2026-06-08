import { describe, expect, it } from 'vitest';
import { ResponseStreamProcessor } from './response-stream';
import { StreamDataExtractor } from '../base';

describe('ResponseStreamProcessor', () => {
  it('accumulates output item and text deltas from response stream events', async () => {
    const extractor = new StreamDataExtractor(
      { content: [], response: undefined as never },
      undefined,
    );

    const processor = new ResponseStreamProcessor('https://example.test/v1');

    await processor.handleEvent(
      {
        type: 'response.output_item.added',
        output_index: 0,
        item: {
          type: 'function_call',
          id: 'call-1',
          call_id: 'call-1',
          name: 'search',
          arguments: '',
          status: 'in_progress',
        },
      } as never,
      extractor,
    );

    await processor.handleEvent(
      {
        type: 'response.function_call_arguments.delta',
        output_index: 0,
        delta: '{"q":"pixi"}',
      } as never,
      extractor,
    );

    await processor.handleEvent(
      {
        type: 'response.output_text.delta',
        output_index: 0,
        content_index: 0,
        delta: 'hello',
      } as never,
      extractor,
    );

    expect(extractor.accumulatedData.content).toHaveLength(1);
    expect(extractor.accumulatedData.content[0]).toMatchObject({
      type: 'function_call',
      name: 'search',
      arguments: '{"q":"pixi"}',
    });
  });
});
