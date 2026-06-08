import { describe, expect, it } from 'vitest';
import { ResponseStreamProcessor } from './response-stream';
import { StreamDataExtractor } from '../base';

describe('ResponseStreamProcessor', () => {
  it('processes a streamed response through the public process API', async () => {
    const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');

    const response = await processor.process(
      (async function* () {
        yield {
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
        } as never;
        yield {
          type: 'response.function_call_arguments.delta',
          output_index: 0,
          delta: '{"q":"pixi"}',
        } as never;
        yield {
          type: 'response.completed',
          response: {
            id: 'resp-1',
            object: 'response',
            created_at: 0,
            status: 'completed',
            error: null,
            incomplete_details: null,
            model: 'gpt-4o-mini',
            output: [
              {
                type: 'function_call',
                id: 'call-1',
                call_id: 'call-1',
                name: 'search',
                arguments: '{"q":"pixi"}',
                status: 'completed',
              },
            ],
            parallel_tool_calls: true,
            temperature: 1,
            tool_choice: 'auto',
            tools: [],
            top_p: 1,
            max_output_tokens: null,
            previous_response_id: null,
            reasoning: null,
            text: {
              format: null,
            },
            truncation: 'disabled',
            usage: null,
          },
        } as never;
      })(),
    );

    expect(response).toMatchObject({
      id: 'resp-1',
      model: 'gpt-4o-mini',
      output: [
        expect.objectContaining({
          type: 'function_call',
          name: 'search',
          arguments: '{"q":"pixi"}',
        }),
      ],
    });
  });

  it('accumulates output item and text deltas from response stream events', async () => {
    const extractor = new StreamDataExtractor(
      { content: [], response: undefined as never },
      undefined,
    );

    const processor = new ResponseStreamProcessor(undefined, 'https://example.test/v1');

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
