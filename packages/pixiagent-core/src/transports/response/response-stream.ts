import type {
  Response,
  ResponseCreateParamsStreaming,
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
import { ResponseApiMessage } from '../../message';
import { DialectResolver, StreamCallbacks, StreamDataExtractor } from '../base';
import { ResponseConversionHelper } from './response-conversion';

export type ResponseStreamAccumulator = StreamDataExtractor<{
  content: Array<ResponseInputItem | ResponseOutputItem>;
  response?: Response;
}>;

export class ResponseStreamProcessor {
  constructor(
    private readonly dialectResolver?: DialectResolver<
      ResponseApiMessage,
      ResponseStreamEvent,
      ResponseCreateParamsStreaming,
      Response
    >,
    private readonly clientBaseUrl?: string,
  ) {}

  async process(
    stream: AsyncIterable<ResponseStreamEvent>,
    callbacks?: StreamCallbacks,
  ): Promise<Response> {
    const streamDataExtractor = new StreamDataExtractor(
      {
        content: Array<ResponseInputItem | ResponseOutputItem>(),
        response: undefined as Response | undefined,
      },
      callbacks,
    );

    for await (const event of stream) {
      await this.applyEvent(event, streamDataExtractor);
    }

    const response = streamDataExtractor.accumulatedData.response;
    if (!response) {
      throw PixiAgentErrorBuilder.modelResponseError(
        'Response stream ended without a terminal response event',
        this.clientBaseUrl,
        'invalid_stream_event',
      );
    }

    return response;
  }

  async handleEvent(
    event: ResponseStreamEvent,
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await this.applyEvent(event, streamDataExtractor);
  }

  private async applyEvent(
    event: ResponseStreamEvent,
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    if (this.dialectResolver) {
      await this.dialectResolver.extractFromDelta('reasoning', event, streamDataExtractor as never);
    }

    switch (event.type) {
      case 'response.created':
      case 'response.queued':
      case 'response.in_progress':
        return;
      case 'error':
        throw this.createResponseError(event as ResponseErrorEvent);
      case 'response.completed':
      case 'response.incomplete':
      case 'response.failed':
        streamDataExtractor.accumulatedData.response = event.response;
        return;
      case 'response.output_item.added':
        await this.applyOutputItemAdded(event, streamDataExtractor);
        return;
      case 'response.output_item.done':
        await this.applyOutputItemDone(event, streamDataExtractor);
        return;
      case 'response.content_part.added':
        await this.applyContentPartAdded(event, streamDataExtractor);
        return;
      case 'response.content_part.done':
        return;
      case 'response.output_text.delta':
        await this.applyOutputTextDelta(event, streamDataExtractor);
        return;
      case 'response.output_text.annotation.added':
        await this.applyOutputTextAnnotationAdded(event, streamDataExtractor);
        return;
      case 'response.output_text.done':
        return;
      case 'response.refusal.delta':
        await this.applyRefusalDelta(event, streamDataExtractor);
        return;
      case 'response.refusal.done':
        return;
      case 'response.reasoning_text.delta':
        await this.applyReasoningTextDelta(event, streamDataExtractor);
        return;
      case 'response.reasoning_text.done':
        return;
      case 'response.reasoning_summary_part.added':
        await this.applyReasoningSummaryPartAdded(event, streamDataExtractor);
        return;
      case 'response.reasoning_summary_text.delta':
        await this.applyReasoningSummaryTextDelta(event, streamDataExtractor);
        return;
      case 'response.reasoning_summary_text.done':
      case 'response.reasoning_summary_part.done':
        return;
      case 'response.function_call_arguments.delta':
        await this.applyFunctionCallArgumentsDelta(event, streamDataExtractor);
        return;
      case 'response.function_call_arguments.done':
        return;
      default:
        return;
    }
  }

  private async applyOutputItemAdded(
    event: ResponseStreamEvent & { type: 'response.output_item.added' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      { key: `output_item_${event.output_index}`, value: event.item },
      (accumulated, data) => {
        if (accumulated.content.length !== event.output_index) {
          throw this.createInvalidEventError('Received output item for non-existing index', event);
        }
        accumulated.content.push(data);
      },
      () => undefined,
      (delta) => (delta.type === 'function_call' ? ResponseConversionHelper.toContentPart(delta) : null),
    );
  }

  private async applyOutputItemDone(
    event: ResponseStreamEvent & { type: 'response.output_item.done' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      { key: `output_item_${event.output_index}`, value: event.item },
      () => undefined,
      (existing, newData) => {
        if (streamDataExtractor.accumulatedData.content.length <= event.output_index) {
          throw this.createInvalidEventError('Received output item done event for non-existing index', event);
        }
        const item = streamDataExtractor.accumulatedData.content[event.output_index];
        if (existing !== item) {
          throw this.createInvalidEventError('Data mismatch for output item between added and done events', event);
        }
        streamDataExtractor.accumulatedData.content[event.output_index] = newData;
      },
      (delta) => {
        if (event.item.type === 'function_call' || event.item.type === 'message' || event.item.type === 'reasoning') {
          return null;
        }
        return ResponseConversionHelper.toContentPart(
          delta as Exclude<ResponseOutputItem, ResponseOutputMessage | ResponseFunctionToolCall>,
        );
      },
    );
  }

  private async applyContentPartAdded(
    event: ResponseStreamEvent & { type: 'response.content_part.added' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      {
        key: `output_item_${event.output_index}_content_${event.content_index}`,
        value: event.part,
      },
      (accumulated, data) => {
        const content = accumulated.content[event.output_index] ?? null;
        if (content == null) {
          throw this.createInvalidEventError('Received content part for non-existing output item', event);
        }

        if (content.type === 'message') {
          const part = data as ResponseOutputText | ResponseOutputRefusal;
          const message = accumulated.content[event.output_index] as ResponseOutputMessage;
          if (message.content.length !== event.content_index) {
            throw this.createInvalidEventError('Received out-of-order content part for message item', event);
          }
          message.content.push(part);
          return;
        }

        if (content.type === 'reasoning') {
          const part = data as { type: 'reasoning_text'; text: string };
          const reasoningItem = accumulated.content[event.output_index] as ResponseReasoningItem;
          reasoningItem.content ??= [];
          reasoningItem.content.push(part);
          return;
        }

        throw this.createInvalidEventError(`Received content part for unsupported item type ${content.type}`, event);
      },
      () => undefined,
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
  }

  private async applyOutputTextDelta(
    event: ResponseStreamEvent & { type: 'response.output_text.delta' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
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
  }

  private async applyOutputTextAnnotationAdded(
    event: ResponseStreamEvent & { type: 'response.output_text.annotation.added' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      {
        key: `output_item_${event.output_index}_content_${event.content_index}`,
        value: { type: 'output_text', text: '', annotations: [event.annotation] } as ResponseOutputText,
      },
      () => undefined,
      (existing, newData) => {
        existing.annotations ??= [];
        existing.annotations.push(...newData.annotations);
      },
      (delta) => ResponseConversionHelper.toContentPart(delta),
    );
  }

  private async applyRefusalDelta(
    event: ResponseStreamEvent & { type: 'response.refusal.delta' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
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
  }

  private async applyReasoningTextDelta(
    event: ResponseStreamEvent & { type: 'response.reasoning_text.delta' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
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
  }

  private async applyReasoningSummaryPartAdded(
    event: ResponseStreamEvent & { type: 'response.reasoning_summary_part.added' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      {
        key: `output_item_${event.output_index}_summary_${event.summary_index}`,
        value: event.part,
      },
      (accumulated, data) => {
        if (accumulated.content.length <= event.output_index) {
          throw this.createInvalidEventError('Received reasoning summary part for non-existing output item', event);
        }
        const content = accumulated.content[event.output_index];
        if (content.type !== 'reasoning') {
          throw this.createInvalidEventError('Received reasoning summary part for a non-reasoning output item', event);
        }
        content.summary ??= [];
        if (content.summary.length !== event.summary_index) {
          throw this.createInvalidEventError('Received out-of-order reasoning summary part', event);
        }
        content.summary.push(data);
      },
      () => undefined,
      (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
    );
  }

  private async applyReasoningSummaryTextDelta(
    event: ResponseStreamEvent & { type: 'response.reasoning_summary_text.delta' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
    await streamDataExtractor.accumulate(
      {
        key: `output_item_${event.output_index}_summary_${event.summary_index}`,
        value: { type: 'summary_text', text: event.delta },
      },
      () => undefined,
      (existing, newData) => {
        existing.text ??= '';
        existing.text += newData.text;
      },
      (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
    );
  }

  private async applyFunctionCallArgumentsDelta(
    event: ResponseStreamEvent & { type: 'response.function_call_arguments.delta' },
    streamDataExtractor: ResponseStreamAccumulator,
  ): Promise<void> {
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
  }

  private createInvalidEventError(message: string, event: ResponseStreamEvent): Error {
    return PixiAgentErrorBuilder.modelResponseError(
      message,
      this.clientBaseUrl,
      'invalid_stream_event',
      { event },
    );
  }

  private createResponseError(event: ResponseErrorEvent): Error {
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
