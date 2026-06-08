import type {
  Response,
  ResponseErrorEvent,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseOutputRefusal,
  ResponseOutputText,
  ResponseReasoningItem,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import { PixiAgentErrorBuilder } from '../../errors';
import { StreamDataExtractor } from '../base';
import { ResponseConversionHelper } from './response-conversion';

export type ResponseStreamAccumulator = StreamDataExtractor<{
  content: Array<ResponseInputItem | ResponseOutputItem>;
  response?: Response;
}>;

export class ResponseStreamProcessor {
  constructor(private readonly clientBaseUrl?: string) {}

  async handleEvent(
    event: ResponseStreamEvent,
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    switch (event.type) {
      case 'response.created':
      case 'response.queued':
      case 'response.in_progress':
        return;
      case 'error':
        throw this.extractResponseError(event as ResponseErrorEvent);
      case 'response.completed':
      case 'response.incomplete':
      case 'response.failed':
        streamDataExtractor.accumulatedData.response = event.response;
        return;
      case 'response.output_item.added':
        await streamDataExtractor.accumulate(
          { key: `output_item_${event.output_index}`, value: event.item },
          (accumulated, data) => {
            if (accumulated.content.length !== event.output_index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received output item for non-existing index ${event.output_index}`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            accumulated.content.push(data);
          },
          (_existing, _newData) => {
            // keyed output item events are appended in the append callback and do not need further merging.
          },
          (delta) => {
            switch (delta.type) {
              case 'function_call':
                return ResponseConversionHelper.toContentPart(delta);
              default:
                return null;
            }
          },
        );
        return;
      case 'response.output_item.done':
        await streamDataExtractor.accumulate(
          { key: `output_item_${event.output_index}`, value: event.item },
          () => undefined,
          (existing, newData) => {
            if (streamDataExtractor.accumulatedData.content.length <= event.output_index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received output item done event for non-existing index ${event.output_index}`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            const item = streamDataExtractor.accumulatedData.content[event.output_index];
            if (existing !== item) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Data mismatch for output item at index ${event.output_index} between added and done events`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            streamDataExtractor.accumulatedData.content[event.output_index] = newData;
          },
          (delta) => {
            switch (event.item.type) {
              case 'function_call':
              case 'message':
              case 'reasoning':
                return null;
              default:
                return ResponseConversionHelper.toContentPart(
                  delta as Exclude<ResponseOutputItem, ResponseOutputMessage | ResponseFunctionToolCall>,
                );
            }
          },
        );
        return;
      case 'response.content_part.added':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_content_${event.content_index}`,
            value: event.part,
          },
          (accumulated, data) => {
            const content =
              accumulated.content.length > event.output_index
                ? accumulated.content[event.output_index]
                : null;
            if (content == null) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received content part for non-existing output item at index ${event.output_index}`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            switch (content.type) {
              case 'message': {
                if (data.type !== 'output_text' && data.type !== 'refusal') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Expected content part of type output_text or refusal for message item, but got ${data.type}`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                    { event },
                  );
                }
                const message = accumulated.content[event.output_index] as ResponseOutputMessage;
                if (message.content.length !== event.content_index) {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received out-of-order content part with content_index ${event.content_index} for message item, expected content_index ${message.content.length}`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                    { event },
                  );
                }
                message.content.push(data);
                break;
              }
              case 'reasoning': {
                if (data.type !== 'reasoning_text') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Expected content part of type reasoning_text for reasoning item, but got ${data.type}`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                    { event },
                  );
                }
                const reasoningItem = accumulated.content[event.output_index] as ResponseReasoningItem;
                if (!reasoningItem.content) {
                  reasoningItem.content = [];
                }
                reasoningItem.content.push(data);
                break;
              }
              default:
                throw PixiAgentErrorBuilder.modelResponseError(
                  `Received content part for unsupported item type ${content.type}`,
                  this.clientBaseUrl,
                  'invalid_stream_event',
                  { event },
                );
            }
          },
          (_existing, _newData) => {
            // keyed content-part events are appended in the append callback and do not need further merging.
          },
          (delta) => {
            switch (delta.type) {
              case 'output_text':
                return delta.text === '' ? null : { type: 'text', text: delta.text };
              case 'refusal':
                return delta.refusal === '' ? null : { type: 'refusal', reason: delta.refusal };
              case 'reasoning_text':
                return delta.text === '' ? null : { type: 'thinking', content: delta.text };
              default:
                return null;
            }
          },
        );
        return;
      case 'response.content_part.done':
        return;
      case 'response.output_text.delta':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_content_${event.content_index}`,
            value: { type: 'output_text', text: event.delta } as ResponseOutputText,
          },
          () => undefined,
          (existing, newData) => {
            existing.text += newData.text;
          },
          (delta) => (delta.text === '' ? null : { type: 'text', text: delta.text }),
        );
        return;
      case 'response.output_text.annotation.added':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_content_${event.content_index}`,
            value: { type: 'output_text', text: '', annotations: [event.annotation] } as ResponseOutputText,
          },
          () => undefined,
          (existing, newData) => {
            if (!existing.annotations) {
              existing.annotations = [];
            }
            existing.annotations.push(...newData.annotations);
          },
          (delta) => ResponseConversionHelper.toContentPart(delta),
        );
        return;
      case 'response.output_text.done':
        return;
      case 'response.refusal.delta':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_content_${event.content_index}`,
            value: { type: 'refusal', refusal: event.delta } as ResponseOutputRefusal,
          },
          () => undefined,
          (existing, newData) => {
            existing.refusal += newData.refusal;
          },
          (delta) => (delta.refusal === '' ? null : ResponseConversionHelper.toContentPart(delta)),
        );
        return;
      case 'response.refusal.done':
        return;
      case 'response.reasoning_text.delta':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_content_${event.content_index}`,
            value: { type: 'reasoning_text', text: event.delta },
          },
          () => undefined,
          (existing, newData) => {
            existing.text += newData.text;
          },
          (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
        );
        return;
      case 'response.reasoning_text.done':
        return;
      case 'response.reasoning_summary_part.added':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_summary_${event.summary_index}`,
            value: event.part,
          },
          (accumulated, data) => {
            if (accumulated.content.length <= event.output_index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received reasoning summary part for non-existing output item at index ${event.output_index}`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            const content = accumulated.content[event.output_index];
            if (content.type !== 'reasoning') {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received reasoning summary part for output item at index ${event.output_index} which is not a reasoning item`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            if (!content.summary) {
              content.summary = [];
            }
            if (content.summary.length !== event.summary_index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received out-of-order reasoning summary part with summary_index ${event.summary_index} for output item at index ${event.output_index}, expected summary_index ${content.summary.length}`,
                this.clientBaseUrl,
                'invalid_stream_event',
                { event },
              );
            }
            content.summary.push(data);
          },
          (_existing, _newData) => {
            // keyed summary events are appended in the append callback and do not need further merging.
          },
          (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
        );
        return;
      case 'response.reasoning_summary_text.delta':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}_summary_${event.summary_index}`,
            value: { type: 'summary_text', text: event.delta },
          },
          () => undefined,
          (existing, newData) => {
            if (!existing.text) {
              existing.text = '';
            }
            existing.text += newData.text;
          },
          (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
        );
        return;
      case 'response.reasoning_summary_text.done':
      case 'response.reasoning_summary_part.done':
        return;
      case 'response.function_call_arguments.delta':
        await streamDataExtractor.accumulate(
          {
            key: `output_item_${event.output_index}`,
            value: { arguments: event.delta, call_id: '', name: '', type: 'function_call' } as ResponseFunctionToolCall,
          },
          () => undefined,
          (existing, newData) => {
            existing.arguments += newData.arguments;
            newData.call_id = existing.call_id;
            newData.name = existing.name;
          },
          (delta) => (delta.arguments === '' ? null : ResponseConversionHelper.toContentPart(delta)),
        );
        return;
      case 'response.function_call_arguments.done':
        return;
      default:
        return;
    }
  }

  private extractResponseError(event: ResponseErrorEvent): Error {
    switch (event.code) {
      case 'rate_limit_exceeded':
      case 'vector_store_timeout':
        return PixiAgentErrorBuilder.modelRequestRetriableError(
          event.message ?? 'Request failed with retriable error',
          this.clientBaseUrl ?? '',
          event.code,
        );
      case 'server_error':
        return PixiAgentErrorBuilder.modelResponseError(
          event.message ?? 'Server error during model response',
          this.clientBaseUrl,
          event.code,
          event,
        );
      default:
        return PixiAgentErrorBuilder.invalidMessage(
          `Error event received from response stream: ${event.message ?? 'Unknown error'}`,
          'assistant',
          event.code ?? undefined,
          event,
        );
    }
  }
}
