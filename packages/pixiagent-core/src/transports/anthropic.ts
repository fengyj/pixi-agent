import Anthropic from '@anthropic-ai/sdk';
import type {
  MessageParam,
  Tool as AnthropicTool,
  TextBlockParam,
  TextCitation,
  TextCitationParam,
  ImageBlockParam,
  DocumentBlockParam,
  ToolUseBlockParam,
  ToolResultBlockParam,
  ThinkingBlockParam,
  RawContentBlockDelta,
  ServerToolUseBlockParam,
} from '@anthropic-ai/sdk/resources/messages';
import type {
  Message,
  MessageStreamParams,
  RawMessageStreamEvent,
} from '@anthropic-ai/sdk/resources/messages/messages';
import {
  ProviderTransport,
  ModelOptions,
  StreamCallbacks,
  DialectResolver,
  ModelRequestOptions,
} from './base';
import {
  AnthropicApiMessage,
  ApiModes,
  SessionMessage,
  ContentPart,
  TextPart,
  ThinkingPart,
  ToolCallPart,
  ToolResultPart,
  ImagePart,
  DocumentPart,
  RefusalPart,
  RoleType,
  CitationFileLocation,
  CitationOthersLocation,
  CitationWebLocation,
  ModelStopReasons,
} from '../message';
import {
  AgentInterruptedError,
  InvalidMessageError,
  ModelRequestTimeoutError,
} from '../errors/types';
import { isLikelyAbortError, isLikelyTimeoutError } from '../errors/guards';

export class AnthropicTransport extends ProviderTransport<AnthropicApiMessage> {
  readonly client: Anthropic;
  private static readonly OFFICIAL_BASE_URL = 'https://api.anthropic.com';
  private readonly configuredBaseUrl?: string;

  private static normalizeBaseUrl(baseUrl?: string): string | undefined {
    if (!baseUrl) return baseUrl;
    const normalized = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    if (normalized.toLowerCase().endsWith('/v1/messages')) {
      return normalized.slice(0, -'/v1/messages'.length);
    }
    return normalized;
  }

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      AnthropicApiMessage,
      RawContentBlockDelta,
      MessageStreamParams,
      Message
    >,
  ) {
    super(ApiModes.ANTHROPIC, dialectResolver);
    this.configuredBaseUrl = AnthropicTransport.normalizeBaseUrl(baseUrl);
    this.client = new Anthropic({ baseURL: this.configuredBaseUrl, apiKey });
  }

  convertFromRawMessage(rawMsg: AnthropicApiMessage): SessionMessage {
    const inner = rawMsg.content;
    const msg = (() => {
      let content: string | ContentPart[] | undefined = undefined;
      let role: RoleType = rawMsg.role;
      if (typeof inner.content === 'string') {
        content = inner.content;
      } else if (Array.isArray(inner.content)) {
        content = inner.content
          .map((block) => {
            if (block.type === 'text') {
              return this.convertToTextPart(block);
            } else if (block.type === 'thinking') {
              return this.convertToThinkingPart(block);
            } else if (block.type === 'image') {
              return this.convertToImagePart(block);
            } else if (block.type === 'document') {
              return this.convertToDocumentPart(block);
            } else if (block.type === 'tool_use' || block.type === 'server_tool_use') {
              return this.convertToToolCallPart(
                block as ToolUseBlockParam | ServerToolUseBlockParam,
              );
            } else if (block.type === 'tool_result') {
              role = 'tool';
              return this.convertToToolResultPart(block);
            } else {
              return null;
            }
          })
          .filter((part) => part !== null);
      }
      return {
        messageId: rawMsg.messageId,
        type: 'session_message',
        role,
        content,
        modelResponseInfo: rawMsg.modelResponseInfo,
        metadata: rawMsg.metadata,
      } as SessionMessage;
    })();
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(msg, rawMsg) : msg;
  }

  convertToRawMessage(msg: SessionMessage): AnthropicApiMessage {
    const inner = (() => {
      return {
        role: msg.role === 'tool' ? 'user' : msg.role,
        content:
          typeof msg.content === 'string'
            ? msg.content
            : msg.content
                .map((part) => {
                  if (part.type === 'text' || part.type === 'refusal') {
                    return this.convertToTextBlockParam(part as TextPart | RefusalPart);
                  } else if (part.type === 'thinking') {
                    return this.convertToThinkingBlockParam(part as ThinkingPart);
                  } else if (part.type === 'image') {
                    return this.convertToImageBlockParam(part as ImagePart);
                  } else if (part.type === 'document') {
                    return this.convertToDocumentBlockParam(part as DocumentPart);
                  } else if (part.type === 'tool_call') {
                    return this.convertToToolUseBlockParam(part as ToolCallPart);
                  } else if (part.type === 'tool_result') {
                    return this.convertToToolResultBlockParam(part as ToolResultPart);
                  } else {
                    return null;
                  }
                })
                .filter((part) => part !== null),
      };
    })() as MessageParam;

    const rawMsg: AnthropicApiMessage = {
      messageId: msg.messageId,
      type: 'anthropic_api_message',
      role: msg.role,
      content: inner,
      modelResponseInfo: msg.modelResponseInfo,
      metadata: msg.metadata,
    };
    return this.dialectResolver ? this.dialectResolver.manipulateRawMessage(rawMsg, msg) : rawMsg;
  }

  // ─── helpers ─────────────────────────────────────────────────────────────

  private getThinkingConfig(
    thinkEffort?: ModelOptions['thinkEffort'],
  ): Anthropic.Messages.ThinkingConfigParam | undefined {
    switch (thinkEffort) {
      case 'disable':
        return { type: 'disabled' };
      case 'low':
        return { type: 'enabled', budget_tokens: 1024, display: 'summarized' };
      case 'medium':
        return { type: 'enabled', budget_tokens: 4096, display: 'summarized' };
      case 'high':
        return { type: 'enabled', budget_tokens: 16000, display: 'summarized' };
      case 'extreme':
        return { type: 'enabled', budget_tokens: 32000, display: 'summarized' };
      default:
        return undefined;
    }
  }

  private getToolChoice(options: ModelOptions): Anthropic.Messages.ToolChoice | undefined {
    if (!options.tools || options.tools.length === 0) return undefined;
    const disableParallel = options.parallelToolCalls === false ? true : undefined;
    if (options.toolChoice === 'force')
      return { type: 'any', disable_parallel_tool_use: disableParallel };
    if (options.toolChoice === 'none') return { type: 'none' };
    if (disableParallel) return { type: 'auto', disable_parallel_tool_use: true };
    return undefined;
  }

  /**
   * Merge consecutive messages with the same role into one.
   * The Anthropic API requires strict user/assistant alternation.
   */
  private mergeConsecutiveSameRoleMessages(messages: MessageParam[]): MessageParam[] {
    const result: MessageParam[] = [];
    for (const msg of messages) {
      const last = result[result.length - 1];
      if (last && last.role === msg.role) {
        const a =
          typeof last.content === 'string'
            ? [{ type: 'text' as const, text: last.content }]
            : [...last.content];
        const b =
          typeof msg.content === 'string'
            ? [{ type: 'text' as const, text: msg.content }]
            : [...msg.content];
        last.content = [...a, ...b];
      } else {
        result.push({ ...msg });
      }
    }
    return result;
  }

  private buildStreamParams(
    options: ModelOptions,
    messages: Array<AnthropicApiMessage>,
  ): MessageStreamParams {
    const mergedMessages = this.mergeConsecutiveSameRoleMessages(messages.map((m) => m.content));

    const tools: AnthropicTool[] | undefined = options.tools?.map(
      (t) =>
        ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }) as AnthropicTool,
    );

    const params: MessageStreamParams = {
      model: options.model,
      messages: mergedMessages,
      system: options.systemPrompt,
      max_tokens: options.maxTokens ?? 4096,
      temperature: options.temperature,
      stop_sequences: options.stopSequences,
      tools: tools && tools.length > 0 ? tools : undefined,
      tool_choice: this.getToolChoice(options),
      thinking: this.getThinkingConfig(options.thinkEffort),
      metadata: options.metadata?.['user_id']
        ? { user_id: String(options.metadata['user_id']) }
        : undefined,
    };

    return this.dialectResolver
      ? (this.dialectResolver.manipulateOptions(options, params) as MessageStreamParams)
      : params;
  }

  private getStreamRequestOptions(requestOptions?: ModelRequestOptions): {
    signal?: AbortSignal;
    timeout?: number;
  } {
    const streamRequestOptions: {
      signal?: AbortSignal;
      timeout?: number;
    } = {};
    if (requestOptions?.signal) {
      streamRequestOptions.signal = requestOptions.signal;
    }
    if (typeof requestOptions?.timeout === 'number') {
      streamRequestOptions.timeout = requestOptions.timeout;
    }
    return streamRequestOptions;
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

  private async generateStream(
    params: MessageStreamParams,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<AnthropicApiMessage, 'messageId'>> {
    let currentToolName: string | undefined;
    const toolInputJsonByIndex = new Map<number, string>();
    let response: Message | undefined;

    const stream = (await this.client.messages.create(
      // Cast because MessageStreamParams can include parser helper types (e.g. output_config null)
      // that are accepted by messages.stream but not by the stricter create(stream:true) overload.
      { ...params, stream: true } as unknown as Anthropic.Messages.MessageCreateParamsStreaming,
      this.getStreamRequestOptions(requestOptions),
    )) as unknown as AsyncIterable<RawMessageStreamEvent>;

    for await (const event of stream) {
      if (event.type === 'message_start') {
        response = structuredClone(event.message);
      } else if (event.type === 'content_block_start') {
        if (event.content_block.type === 'tool_use') {
          currentToolName = event.content_block.name;
        }
        if (response && Array.isArray(response.content)) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (response.content as any[])[event.index] = structuredClone(event.content_block);
        }
      } else if (event.type === 'content_block_delta') {
        const delta = event.delta;
        if (delta.type === 'text_delta') {
          callbacks?.onTextChunk?.(delta.text);

          if (response && Array.isArray(response.content)) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const block = (response.content as any[])[event.index];
            if (block?.type === 'text') {
              block.text = `${block.text ?? ''}${delta.text}`;
            }
          }
        } else if (delta.type === 'thinking_delta') {
          callbacks?.onThinkingChunk?.(delta.thinking);

          if (response && Array.isArray(response.content)) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const block = (response.content as any[])[event.index];
            if (block?.type === 'thinking') {
              block.thinking = `${block.thinking ?? ''}${delta.thinking}`;
            }
          }
        } else if (delta.type === 'input_json_delta') {
          if (currentToolName) {
            // onToolUse is removed from StreamCallbacks.
          }
          toolInputJsonByIndex.set(
            event.index,
            `${toolInputJsonByIndex.get(event.index) ?? ''}${delta.partial_json}`,
          );
        }
      } else if (event.type === 'content_block_stop') {
        currentToolName = undefined;
      } else if (event.type === 'message_delta') {
        if (response) {
          response.stop_reason = event.delta.stop_reason;
          response.stop_sequence = event.delta.stop_sequence;
          response.stop_details = event.delta.stop_details;
          response.usage = {
            ...response.usage,
            ...event.usage,
            input_tokens: event.usage.input_tokens ?? response.usage.input_tokens,
            cache_creation_input_tokens:
              event.usage.cache_creation_input_tokens ?? response.usage.cache_creation_input_tokens,
            cache_read_input_tokens:
              event.usage.cache_read_input_tokens ?? response.usage.cache_read_input_tokens,
          } as Message['usage'];
        }
      } else if (event.type === 'message_stop') {
        // Nothing to do at stream end for simplified StreamCallbacks.
      }
    }

    if (!response) {
      throw new Error('Anthropic stream ended without a message_start event');
    }

    if (Array.isArray(response.content)) {
      for (const [index, partialJson] of toolInputJsonByIndex.entries()) {
        const block = response.content[index];
        if (block?.type === 'tool_use' && partialJson.length > 0) {
          try {
            block.input = JSON.parse(partialJson);
          } catch {
            block.input = partialJson; // fallback to raw string if JSON parsing fails
          }
        }
      }
    }

    return this.toModelResponse(response);
  }

  async generate(
    options: ModelOptions,
    messages: Array<AnthropicApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<AnthropicApiMessage, 'messageId'>> {
    const params = this.buildStreamParams(options, messages);

    try {
      return await this.generateStream(params, callbacks, requestOptions);
    } catch (error) {
      const mappedError = this.wrapRequestError(error, requestOptions);
      throw mappedError;
    }
  }

  private wrapRequestError(error: unknown, requestOptions?: ModelRequestOptions): unknown {
    const signal = requestOptions?.signal;
    if (signal?.aborted) {
      const reason = signal.reason;
      if (reason instanceof AgentInterruptedError) {
        return reason;
      }
      if (typeof reason === 'string' && reason.length > 0) {
        return new AgentInterruptedError(reason);
      }
      if (reason instanceof Error && reason.message) {
        return new AgentInterruptedError(reason.message);
      }
      return new AgentInterruptedError();
    }
    if (isLikelyAbortError(error)) {
      return new AgentInterruptedError();
    }
    if (!isLikelyTimeoutError(error)) {
      return error;
    }
    return new ModelRequestTimeoutError(this.client.baseURL, requestOptions?.timeout, error);
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

  private convertToTextBlockParam(part: TextPart | RefusalPart): TextBlockParam {
    const textBlock: TextBlockParam = {
      type: 'text',
      text: part.type === 'text' ? part.text : part.reason,
    };

    if (part.type === 'text' && part.citations && part.citations.length > 0) {
      const citations = part.citations
        .map((citation) => this.convertCitationToAnthropicTextCitation(citation))
        .filter((citation): citation is TextCitation => citation !== undefined);
      if (citations.length > 0) {
        textBlock.citations = citations;
      }
    }

    return textBlock;
  }

  private convertCitationToAnthropicTextCitation(
    citation: CitationFileLocation | CitationOthersLocation | CitationWebLocation,
  ): TextCitation | undefined {
    if (citation.type === 'web_location') {
      const extra = citation.extra ?? {};
      return {
        type: 'web_search_result_location',
        cited_text: citation.citedText,
        encrypted_index: String(extra.encrypted_index ?? citation.url),
        title: citation.title ?? null,
        url: citation.url,
      };
    }

    if (citation.type === 'file_location') {
      const extra = citation.extra ?? {};
      switch (citation.rawCitationType) {
        case 'char_location':
          return {
            type: 'char_location',
            cited_text: citation.citedText,
            document_index: Number(extra.document_index ?? 0),
            document_title: typeof extra.document_title === 'string' ? extra.document_title : null,
            file_id: typeof extra.file_id === 'string' ? extra.file_id : null,
            start_char_index: citation.startIndex ?? 0,
            end_char_index: citation.endIndex ?? 0,
          };
        case 'page_location':
          return {
            type: 'page_location',
            cited_text: citation.citedText,
            document_index: Number(extra.document_index ?? 0),
            document_title: typeof extra.document_title === 'string' ? extra.document_title : null,
            file_id: typeof extra.file_id === 'string' ? extra.file_id : null,
            start_page_number: citation.startIndex ?? 0,
            end_page_number: citation.endIndex ?? 0,
          };
        case 'content_block_location':
          return {
            type: 'content_block_location',
            cited_text: citation.citedText,
            document_index: Number(extra.document_index ?? 0),
            document_title: typeof extra.document_title === 'string' ? extra.document_title : null,
            file_id: typeof extra.file_id === 'string' ? extra.file_id : null,
            start_block_index: citation.startIndex ?? 0,
            end_block_index: citation.endIndex ?? 0,
          };
        default:
          return undefined;
      }
    }

    if (citation.type === 'others_location') {
      const extra = citation.extra ?? {};
      if (citation.rawCitationType === 'web_search_result_location') {
        return {
          type: 'web_search_result_location',
          cited_text: citation.citedText,
          encrypted_index: String(extra.encrypted_index ?? citation.source),
          title: citation.title ?? null,
          url: String(extra.url ?? citation.source),
        };
      }
      return {
        type: 'search_result_location',
        cited_text: citation.citedText,
        source: citation.source,
        search_result_index: Number(extra.search_result_index ?? 0),
        start_block_index: citation.startIndex ?? 0,
        end_block_index: citation.endIndex ?? 0,
        title: citation.title ?? null,
      };
    }

    return undefined;
  }

  private convertToThinkingBlockParam(part: ThinkingPart): ThinkingBlockParam {
    return { type: 'thinking', thinking: part.content, signature: '' };
  }

  private convertToImageBlockParam(part: ImagePart): ImageBlockParam | null {
    const img = part.image;
    if (img.sourceType === 'base64') {
      if (
        img.mimeType === 'image/jpeg' ||
        img.mimeType === 'image/png' ||
        img.mimeType === 'image/gif' ||
        img.mimeType === 'image/webp'
      ) {
        return {
          type: 'image',
          source: { type: 'base64', media_type: img.mimeType, data: img.data },
        };
      }
    } else if (img.sourceType === 'url') {
      return { type: 'image', source: { type: 'url', url: img.url } };
    }
    return null;
  }

  private convertToDocumentBlockParam(part: DocumentPart): DocumentBlockParam | null {
    const doc = part.document;
    if (doc.sourceType === 'base64') {
      if (doc.mimeType === 'text/plain') {
        return {
          type: 'document',
          source: { type: 'text', media_type: 'text/plain', data: doc.data! },
        };
      } else if (doc.mimeType === 'application/pdf') {
        return {
          type: 'document',
          source: { type: 'base64', media_type: 'application/pdf', data: doc.data! },
        };
      }
    } else if (doc.sourceType === 'url') {
      return { type: 'document', source: { type: 'url', url: doc.url! } };
    }
    return null;
  }

  private convertToToolUseBlockParam(part: ToolCallPart): ToolUseBlockParam {
    try {
      return {
        type: 'tool_use',
        id: part.id,
        name: part.name,
        input: JSON.parse(part.arguments || '{}'),
      };
    } catch {
      return {
        type: 'tool_use',
        id: part.id,
        name: part.name,
        input: part.arguments,
      };
    }
  }

  private convertToToolResultBlockParam(part: ToolResultPart): ToolResultBlockParam {
    if (typeof part.result === 'string') {
      return {
        type: 'tool_result',
        tool_use_id: part.id,
        content: part.result,
        is_error: part.isError ?? undefined,
      };
    } else if (Array.isArray(part.result)) {
      return {
        type: 'tool_result',
        tool_use_id: part.id,
        content: part.result
          .map((p) => {
            if (p.type === 'text') {
              return this.convertToTextBlockParam(p as TextPart);
            } else if (p.type === 'image') {
              return this.convertToImageBlockParam(p as ImagePart);
            } else if (p.type === 'document') {
              return this.convertToDocumentBlockParam(p as DocumentPart);
            }
            // skip unrecognized or unsupported content parts
            return null;
          })
          .filter((b): b is TextBlockParam | ImageBlockParam | DocumentBlockParam => b !== null),
        is_error: part.isError ?? undefined,
      };
    } else {
      throw new InvalidMessageError(
        'ToolResultPart result must be a string or an array of ContentParts.',
      );
    }
  }

  convertToTextPart(block: TextBlockParam): TextPart {
    return {
      type: 'text',
      text: block.text,
      citations:
        block.citations?.flatMap((c) => {
          const citation = this.convertAnthropicCitationToSessionCitation(c);
          return citation ? [citation] : [];
        }) ?? undefined,
    };
  }

  private convertAnthropicCitationToSessionCitation(
    citation: TextCitation | TextCitationParam,
  ): CitationFileLocation | CitationOthersLocation | CitationWebLocation | null {
    const extra: Record<string, unknown> = {
      ...(citation as unknown as Record<string, unknown>),
    };
    delete extra.type;
    delete extra.cited_text;

    if (citation.type === 'char_location') {
      return {
        type: 'file_location',
        fileName:
          typeof citation.document_title === 'string'
            ? citation.document_title
            : String('file_id' in citation && citation.file_id ? citation.file_id : ''),
        citedText: citation.cited_text,
        title: typeof citation.document_title === 'string' ? citation.document_title : undefined,
        startIndex: citation.start_char_index,
        endIndex: citation.end_char_index,
        rawCitationType: citation.type,
        extra,
      };
    }

    if (citation.type === 'page_location') {
      return {
        type: 'file_location',
        fileName:
          typeof citation.document_title === 'string'
            ? citation.document_title
            : String('file_id' in citation && citation.file_id ? citation.file_id : ''),
        citedText: citation.cited_text,
        title: typeof citation.document_title === 'string' ? citation.document_title : undefined,
        startIndex: citation.start_page_number,
        endIndex: citation.end_page_number,
        rawCitationType: citation.type,
        extra,
      };
    }

    if (citation.type === 'content_block_location') {
      return {
        type: 'file_location',
        fileName:
          typeof citation.document_title === 'string'
            ? citation.document_title
            : String('file_id' in citation && citation.file_id ? citation.file_id : ''),
        citedText: citation.cited_text,
        title: typeof citation.document_title === 'string' ? citation.document_title : undefined,
        startIndex: citation.start_block_index,
        endIndex: citation.end_block_index,
        rawCitationType: citation.type,
        extra,
      };
    }

    if (citation.type === 'web_search_result_location') {
      return {
        type: 'others_location',
        source: citation.url,
        citedText: citation.cited_text,
        title: citation.title ?? undefined,
        startIndex: undefined,
        endIndex: undefined,
        rawCitationType: citation.type,
        extra,
      };
    }

    if (citation.type === 'search_result_location') {
      return {
        type: 'others_location',
        source: citation.source,
        citedText: citation.cited_text,
        title: citation.title ?? undefined,
        startIndex: citation.start_block_index,
        endIndex: citation.end_block_index,
        rawCitationType: citation.type,
        extra,
      };
    }

    return null;
  }

  convertToThinkingPart(block: ThinkingBlockParam): ThinkingPart {
    return { type: 'thinking', content: block.thinking };
  }

  convertToImagePart(block: ImageBlockParam): ImagePart {
    if (block.source.type === 'base64') {
      return {
        type: 'image',
        image: { sourceType: 'base64', mimeType: block.source.media_type, data: block.source.data },
      };
    } else if (block.source.type === 'url') {
      return { type: 'image', image: { sourceType: 'url', url: block.source.url } };
    }
    throw new InvalidMessageError('Unsupported ImageBlockParam source type.');
  }

  convertToDocumentPart(block: DocumentBlockParam): DocumentPart | TextPart {
    if (block.source.type === 'base64') {
      return {
        type: 'document',
        document: {
          sourceType: 'base64',
          mimeType: block.source.media_type,
          data: block.source.data,
        },
      };
    } else if (block.source.type === 'text') {
      return {
        type: 'text',
        text: block.source.data,
      };
    } else if (block.source.type === 'url') {
      return { type: 'document', document: { sourceType: 'url', url: block.source.url } };
    }
    throw new InvalidMessageError('Unsupported DocumentBlockParam source type.');
  }

  convertToToolCallPart(block: ToolUseBlockParam | ServerToolUseBlockParam): ToolCallPart {
    if (block.type === 'server_tool_use') {
      return {
        type: 'tool_call',
        id: block.id,
        name: block.name,
        arguments: ContentPart.createProviderToolCallArguments('anthropic', block.type, block),
      };
    }

    return {
      type: 'tool_call',
      id: block.id,
      name: block.name,
      arguments: JSON.stringify(block.input),
    };
  }

  convertToToolResultPart(block: ToolResultBlockParam): ToolResultPart {
    let result: string | Array<TextPart | ImagePart | DocumentPart> = '';
    if (typeof block.content === 'string') {
      result = block.content;
    } else if (Array.isArray(block.content)) {
      result = block.content.map((b) => {
        switch (b.type) {
          case 'text':
            return this.convertToTextPart(b);
          case 'image':
            return this.convertToImagePart(b);
          case 'document':
            return this.convertToDocumentPart(b);
          default:
            throw new InvalidMessageError(
              `Unsupported content block type in tool result: ${b.type}`,
            );
        }
      });
    }
    return {
      type: 'tool_result',
      id: block.tool_use_id,
      result,
      isError: block.is_error ?? undefined,
    };
  }
}
