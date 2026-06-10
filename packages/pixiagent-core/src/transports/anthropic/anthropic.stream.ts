import type {
  Message,
  MessageStreamEvent,
  RawContentBlockDelta,
  MessageCreateParamsStreaming,
  ToolUseBlock,
} from '@anthropic-ai/sdk/resources/messages/messages';
import type { ContentBlock } from '@anthropic-ai/sdk/resources/messages';
import { PixiAgentErrorBuilder } from '../../errors';
import { AnthropicApiMessage, ContentPart, ModelStopReasons, TextPart, ToolCallPart, ServerToolUsePart } from '../../message';
import { assertNever } from '../../utils';
import { DialectResolver, StreamCallbacks, StreamDataExtractor } from '../base';
import { AnthropicMessageConverter } from './anthropic.converters';

export class AnthropicStreamProcessor {
  constructor(
    private readonly dialectResolver?: DialectResolver<
      AnthropicApiMessage,
      RawContentBlockDelta,
      MessageCreateParamsStreaming,
      Message
    >,
    private readonly clientBaseUrl?: string,
  ) {}

  async process(
    stream: AsyncIterable<MessageStreamEvent>,
    callbacks?: StreamCallbacks,
  ): Promise<Omit<AnthropicApiMessage, 'messageId'>> {
    const streamDataExtractor = new StreamDataExtractor(
      {
        content: Array<ContentBlock>(),
        id: '',
        container: null,
        model: '',
        role: 'assistant',
        stop_details: null,
        stop_reason: null,
        stop_sequence: null,
        type: 'message',
        usage: {
          input_tokens: 0,
          output_tokens: 0,
        },
      } as Message,
      callbacks,
    );

    for await (const event of stream) {
      await this.handleEvent(event, streamDataExtractor);
    }

    return this.toModelResponse(streamDataExtractor.accumulatedData);
  }

  async handleEvent(
    event: MessageStreamEvent,
    streamDataExtractor: StreamDataExtractor<Message>,
  ): Promise<void> {
    switch (event.type) {
      case 'message_start':
        streamDataExtractor.accumulatedData.container = event.message.container;
        streamDataExtractor.accumulatedData.model = event.message.model;
        streamDataExtractor.accumulatedData.role = event.message.role;
        streamDataExtractor.accumulatedData.id = event.message.id;
        streamDataExtractor.accumulatedData.stop_details = event.message.stop_details;
        streamDataExtractor.accumulatedData.stop_reason = event.message.stop_reason;
        streamDataExtractor.accumulatedData.stop_sequence = event.message.stop_sequence;
        streamDataExtractor.accumulatedData.usage = event.message.usage;
        streamDataExtractor.accumulatedData.content.push(...event.message.content);
        return;

      case 'message_delta':
        if (event.delta.stop_reason) {
          streamDataExtractor.accumulatedData.stop_reason = event.delta.stop_reason;
        }
        if (event.delta.container) {
          streamDataExtractor.accumulatedData.container = event.delta.container;
        }
        if (event.delta.stop_details) {
          streamDataExtractor.accumulatedData.stop_details = event.delta.stop_details;
        }
        if (event.delta.stop_sequence) {
          streamDataExtractor.accumulatedData.stop_sequence = event.delta.stop_sequence;
        }
        if (event.usage.cache_creation_input_tokens) {
          streamDataExtractor.accumulatedData.usage.cache_creation_input_tokens =
            event.usage.cache_creation_input_tokens;
        }
        if (event.usage.cache_read_input_tokens) {
          streamDataExtractor.accumulatedData.usage.cache_read_input_tokens =
            event.usage.cache_read_input_tokens;
        }
        if (event.usage.input_tokens) {
          streamDataExtractor.accumulatedData.usage.input_tokens = event.usage.input_tokens;
        }
        if (event.usage.output_tokens) {
          streamDataExtractor.accumulatedData.usage.output_tokens = event.usage.output_tokens;
        }
        if (event.usage.server_tool_use) {
          streamDataExtractor.accumulatedData.usage.server_tool_use =
            event.usage.server_tool_use;
        }
        return;

      case 'content_block_start':
        await streamDataExtractor.accumulate(
          { key: `content_${event.index}`, value: event.content_block },
          (accumulatedData, newData) => {
            if (accumulatedData.content.length !== event.index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received content block index ${event.index} does not match expected index ${accumulatedData.content.length}.`,
                this.clientBaseUrl,
                'invalid_stream_event',
              );
            }
            accumulatedData.content.push(newData);
          },
          () => undefined,
          (data) => this.toContentPart(data),
        );
        return;

      case 'content_block_delta':
        await streamDataExtractor.accumulate(
          { key: `content_${event.index}`, value: event.delta },
          () => undefined,
          (_existing, newData, accumulatedData) => {
            if (accumulatedData.content.length <= event.index) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Received content block delta for index ${event.index} which exceeds current content length ${accumulatedData.content.length}.`,
                this.clientBaseUrl,
                'invalid_stream_event',
              );
            }
            const block = accumulatedData.content[event.index];
            switch (newData.type) {
              case 'text_delta':
                if (block.type !== 'text') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received text_delta for content block at index ${event.index} which is not of type 'text'.`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                  );
                }
                block.text = `${block.text ?? ''}${newData.text}`;
                return;
              case 'citations_delta':
                if (block.type !== 'text') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received citations_delta for content block at index ${event.index} which is not of type 'text'.`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                  );
                }
                block.citations ??= [];
                block.citations.push(newData.citation);
                return;
              case 'thinking_delta':
                if (block.type !== 'thinking') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received thinking_delta for content block at index ${event.index} which is not of type 'thinking'.`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                  );
                }
                block.thinking = `${block.thinking ?? ''}${newData.thinking}`;
                return;
              case 'signature_delta':
                if (block.type !== 'thinking') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received signature_delta for content block at index ${event.index} which is not of type 'thinking'.`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                  );
                }
                block.signature = `${block.signature ?? ''}${newData.signature}`;
                return;
              case 'input_json_delta':
                if (block.type !== 'tool_use') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received input_json_delta for content block at index ${event.index} which is not of type 'tool_use'.`,
                    this.clientBaseUrl,
                    'invalid_stream_event',
                  );
                }
                block.input = `${block.input ?? ''}${newData.partial_json}`;
                return;
              default:
                assertNever(newData);
            }
          },
          (delta) => this.toDeltaContentPart(delta, streamDataExtractor, event.index),
        );
        return;

      case 'content_block_stop': {
        if (event.index >= streamDataExtractor.accumulatedData.content.length) {
          throw PixiAgentErrorBuilder.modelResponseError(
            `Received content block stop for index ${event.index} which exceeds current content length ${streamDataExtractor.accumulatedData.content.length}.`,
            this.clientBaseUrl,
            'invalid_stream_event',
          );
        }
        const block = streamDataExtractor.accumulatedData.content[event.index];
        if (block.type === 'tool_use' && typeof block.input === 'string') {
          try {
            block.input = JSON.parse(block.input as string);
          } catch (error) {
            throw PixiAgentErrorBuilder.modelResponseError(
              `Failed to parse input JSON for content block at index ${event.index}: ${(error as Error).message}`,
              this.clientBaseUrl,
              'invalid_stream_event',
              error,
            );
          }
        }
        return;
      }

      case 'message_stop':
        return;

      default:
        assertNever(event as never);
    }
  }

  private toContentPart(data: unknown): ContentPart | null {
    if (!data || typeof data !== 'object' || !('type' in data)) {
      return null;
    }

    switch (data.type) {
      case 'text':
      case 'thinking':
      case 'tool_use':
      case 'bash_code_execution_tool_result':
      case 'code_execution_tool_result':
      case 'container_upload':
      case 'redacted_thinking':
      case 'server_tool_use':
      case 'text_editor_code_execution_tool_result':
      case 'tool_search_tool_result':
      case 'web_fetch_tool_result':
      case 'web_search_tool_result': {
        const parts = AnthropicMessageConverter.toParts(data as never);
        return Array.isArray(parts) ? (parts[0] ?? null) : (parts as ContentPart);
      }
      default:
        return assertNever(data as never);
    }
  }

  private toDeltaContentPart(
    delta: RawContentBlockDelta,
    streamDataExtractor: StreamDataExtractor<Message>,
    index: number,
  ): ContentPart | null {
    if (!delta || typeof delta !== 'object' || !('type' in delta)) {
      return null;
    }

    switch (delta.type) {
      case 'text_delta':
        return { type: 'text', text: delta.text } as TextPart;
      case 'citations_delta': {
        const parts = AnthropicMessageConverter.toParts({
          type: 'text',
          text: '',
          citations: [delta.citation],
        } as never);
        return Array.isArray(parts) ? (parts[0] ?? null) : (parts as TextPart);
      }
      case 'thinking_delta':
        return { type: 'thinking', content: delta.thinking } as ContentPart;
      case 'input_json_delta': {
        const block = streamDataExtractor.accumulatedData.content[index] as ToolUseBlock;
        const parts = AnthropicMessageConverter.toParts({
          ...block,
          input: delta.partial_json,
        } as never);
        return Array.isArray(parts)
          ? (parts[0] ?? null)
          : (parts as ToolCallPart | ServerToolUsePart);
      }
      case 'signature_delta':
        return null;
      default:
        return assertNever(delta as never);
    }
  }

  private toModelResponse(response: Message): Omit<AnthropicApiMessage, 'messageId'> {
    return {
      type: 'anthropic_api_message',
      role: 'assistant',
      content: {
        role: response.role,
        content: response.content,
      },
      modelResponseInfo: {
        responseId: response.id,
        responseModel: response.model,
        stopReason: this.getStopReason(
          this.dialectResolver?.extractFromResponse('stop_reason', response) ??
            (response.stop_reason as string),
        ),
        refusal: response.stop_details?.explanation ?? undefined,
        usage: {
          inputTokens: response.usage.input_tokens,
          outputTokens: response.usage.output_tokens,
          reasoningTokens:
            this.dialectResolver?.extractFromResponse('reasoning_tokens', response) ?? undefined,
          cacheCreatedTokens:
            this.dialectResolver?.extractFromResponse('cache_created_tokens', response) ??
            response.usage.cache_creation_input_tokens ??
            undefined,
          cacheReadTokens:
            this.dialectResolver?.extractFromResponse('cache_read_tokens', response) ??
            response.usage.cache_read_input_tokens ??
            undefined,
          totalTokens: response.usage.input_tokens + response.usage.output_tokens,
        },
      },
      metadata: {
        pixiagent_response_id: response.id,
        pixiagent_response_stop_details: response.stop_details,
        pixiagent_response_stop_reason: response.stop_reason,
        pixiagent_response_stop_sequence: response.stop_sequence,
      },
    };
  }

  private getStopReason(finishReason: string): ModelStopReasons {
    switch (finishReason) {
      case 'end_turn':
      case 'stop_sequence':
      case 'pause_turn':
        return ModelStopReasons.STOP;
      case 'tool_use':
        return ModelStopReasons.TOOL_CALL;
      case 'max_tokens':
        return ModelStopReasons.MAX_TOKENS;
      case 'refusal':
        return ModelStopReasons.REFUSAL;
      default:
        return ModelStopReasons.OTHERS;
    }
  }
}
