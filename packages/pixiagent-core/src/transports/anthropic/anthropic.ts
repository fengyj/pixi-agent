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
  BashCodeExecutionToolResultBlock,
  BashCodeExecutionToolResultBlockParam,
  CodeExecutionToolResultBlock,
  CodeExecutionToolResultBlockParam,
  ContainerUploadBlock,
  ContainerUploadBlockParam,
  ContentBlock,
  ContentBlockParam,
  Message,
  MessageCreateParamsStreaming,
  SearchResultBlockParam,
  ServerToolUseBlock,
  TextBlock,
  TextEditorCodeExecutionToolResultBlock,
  TextEditorCodeExecutionToolResultBlockParam,
  ThinkingBlock,
  ToolReferenceBlock,
  ToolReferenceBlockParam,
  ToolSearchToolResultBlock,
  ToolSearchToolResultBlockParam,
  ToolUseBlock,
  WebFetchToolResultBlock,
  WebFetchToolResultBlockParam,
  WebSearchToolResultBlock,
  WebSearchToolResultBlockParam,
} from '@anthropic-ai/sdk/resources/messages/messages';
import {
  ProviderTransport,
  ModelOptions,
  StreamCallbacks,
  DialectResolver,
  ModelRequestOptions,
  StreamDataExtractor,
} from '../base';
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
  CitationFileLocation,
  CitationOthersLocation,
  CitationWebLocation,
  ModelStopReasons,
  ServerToolUsePart,
  AudioPart,
  VideoPart,
  Citation,
} from '../../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../../errors';
import { assertNever, ContentParts } from '../../utils';

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
      MessageCreateParamsStreaming,
      Message
    >,
  ) {
    super(ApiModes.ANTHROPIC, dialectResolver);
    this.configuredBaseUrl = AnthropicTransport.normalizeBaseUrl(baseUrl);
    this.client = new Anthropic({ baseURL: this.configuredBaseUrl, apiKey });
  }

  convertFromRawMessage(rawMsg: AnthropicApiMessage): SessionMessage {
    const inner = rawMsg.content;
    const msg: SessionMessage = {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: rawMsg.role,
      content:
        typeof inner.content === 'string'
          ? inner.content
          : inner.content.map((block) => ConvertHelper.toParts(block)).flat(),
      modelResponseInfo: rawMsg.modelResponseInfo,
      metadata: rawMsg.metadata,
    };
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(msg, rawMsg) : msg;
  }

  convertToRawMessage(msg: SessionMessage): AnthropicApiMessage {
    const content = {
      role: msg.role === 'tool' ? 'user' : msg.role,
      content:
        typeof msg.content === 'string'
          ? msg.content
          : msg.content
              .map((part) => ConvertHelper.toBlockParam(part))
              .filter((part) => part !== null),
    };

    const rawMsg: AnthropicApiMessage = {
      messageId: msg.messageId,
      type: 'anthropic_api_message',
      role: msg.role,
      content: content,
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
  ): MessageCreateParamsStreaming {
    const mergedMessages = this.mergeConsecutiveSameRoleMessages(messages.map((m) => m.content));

    const tools: AnthropicTool[] | undefined = options.tools?.map(
      (t) =>
        ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }) as AnthropicTool,
    );

    const params: MessageCreateParamsStreaming = {
      stream: true,
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
      ? (this.dialectResolver.manipulateOptions(options, params) as MessageCreateParamsStreaming)
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
    params: MessageCreateParamsStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<AnthropicApiMessage, 'messageId'>> {
    const stream = await this.client.messages.create(
      // Cast because MessageStreamParams can include parser helper types (e.g. output_config null)
      // that are accepted by messages.stream but not by the stricter create(stream:true) overload.
      params,
      this.getStreamRequestOptions(requestOptions),
    );

    const streamDataExtractor = new StreamDataExtractor(
      {
        content: Array<ContentBlock>(),
        id: '',
        container: null,
        model: '',
        role: 'assistant',
        stop_details: null,
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
      switch (event.type) {
        case 'message_start':
          streamDataExtractor.accumulatedData.container = event.message.container;
          streamDataExtractor.accumulatedData.model = event.message.model;
          streamDataExtractor.accumulatedData.role = event.message.role;
          streamDataExtractor.accumulatedData.id = event.message.id;
          streamDataExtractor.accumulatedData.stop_details = event.message.stop_details;
          streamDataExtractor.accumulatedData.stop_sequence = event.message.stop_sequence;
          streamDataExtractor.accumulatedData.usage = event.message.usage;
          streamDataExtractor.accumulatedData.content = event.message.content;
          break;
        case 'message_delta':
          if (!event.delta.container)
            streamDataExtractor.accumulatedData.container = event.delta.container;
          if (!event.delta.stop_details)
            streamDataExtractor.accumulatedData.stop_details = event.delta.stop_details;
          if (!event.delta.stop_sequence)
            streamDataExtractor.accumulatedData.stop_sequence = event.delta.stop_sequence;
          if (event.usage.cache_creation_input_tokens)
            streamDataExtractor.accumulatedData.usage.cache_creation_input_tokens =
              event.usage.cache_creation_input_tokens;
          if (event.usage.cache_read_input_tokens)
            streamDataExtractor.accumulatedData.usage.cache_read_input_tokens =
              event.usage.cache_read_input_tokens;
          if (event.usage.input_tokens)
            streamDataExtractor.accumulatedData.usage.input_tokens = event.usage.input_tokens;
          if (event.usage.output_tokens)
            streamDataExtractor.accumulatedData.usage.output_tokens = event.usage.output_tokens;
          if (event.usage.server_tool_use)
            streamDataExtractor.accumulatedData.usage.server_tool_use = event.usage.server_tool_use;
          break;
        case 'content_block_start':
          streamDataExtractor.accumulate(
            {
              key: `content_${event.index}`,
              value: event.content_block,
            },
            (accumulatedData, newData) => {
              if (accumulatedData.content.length !== event.index) {
                throw PixiAgentErrorBuilder.modelResponseError(
                  `Received content block index ${event.index} does not match expected index ${accumulatedData.content.length}.`,
                );
              }
              accumulatedData.content.push(newData);
            },
            (_accumulatedData, _newData) => {},
            (data) => {
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
                case 'web_search_tool_result':
                  return ConvertHelper.toParts(data) as ContentPart;
                default:
                  assertNever(data);
              }
            },
          );
          break;
        case 'content_block_delta':
          streamDataExtractor.accumulate(
            {
              key: `content_${event.index}`,
              value: event.delta,
            },
            (_accumulatedData, _newData) => {},
            (_existing, newData, accumulatedData) => {
              if (accumulatedData.content.length <= event.index) {
                throw PixiAgentErrorBuilder.modelResponseError(
                  `Received content block delta for index ${event.index} which exceeds current content length ${accumulatedData.content.length}.`,
                );
              }
              const block = accumulatedData.content[event.index];
              switch (newData.type) {
                case 'text_delta':
                  if (block.type !== 'text')
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received text_delta for content block at index ${event.index} which is not of type 'text'.`,
                    );
                  block.text = `${block.text ?? ''}${newData.text}`;
                  break;
                case 'citations_delta':
                  if (block.type !== 'text')
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received citations_delta for content block at index ${event.index} which is not of type 'text'.`,
                    );
                  block.citations ??= [];
                  block.citations.push(newData.citation);
                  break;
                case 'thinking_delta':
                  if (block.type !== 'thinking')
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received thinking_delta for content block at index ${event.index} which is not of type 'thinking'.`,
                    );
                  block.thinking = `${block.thinking ?? ''}${newData.thinking}`;
                  break;
                case 'signature_delta':
                  if (block.type !== 'thinking')
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received signature_delta for content block at index ${event.index} which is not of type 'thinking'.`,
                    );
                  block.signature = `${block.signature ?? ''}${newData.signature}`;
                  break;
                case 'input_json_delta':
                  if (block.type !== 'tool_use')
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received input_json_delta for content block at index ${event.index} which is not of type 'tool_use'.`,
                    );
                  block.input = `${block.input ?? ''}${newData.partial_json}`;
                  break;
                default:
                  assertNever(newData);
              }
            },
            (delta) => {
              switch (delta.type) {
                case 'text_delta':
                  return {
                    type: 'text',
                    text: delta.text,
                  };
                case 'citations_delta':
                  return ConvertHelper.toParts({
                    type: 'text',
                    text: '',
                    citations: [delta.citation],
                  }) as TextPart;
                case 'thinking_delta':
                  return {
                    type: 'thinking',
                    content: delta.thinking,
                  };
                case 'input_json_delta': {
                  const block = streamDataExtractor.accumulatedData.content[
                    event.index
                  ] as ToolUseBlock;
                  return ConvertHelper.toParts({
                    ...block,
                    input: delta.partial_json,
                  }) as ToolCallPart | ServerToolUsePart;
                }
                case 'signature_delta':
                  return null;
                default:
                  assertNever(delta);
              }
            },
          );
          break;
        case 'content_block_stop': {
          if (event.index >= streamDataExtractor.accumulatedData.content.length) {
            throw PixiAgentErrorBuilder.modelResponseError(
              `Received content block stop for index ${event.index} which exceeds current content length ${streamDataExtractor.accumulatedData.content.length}.`,
            );
          }
          const block = streamDataExtractor.accumulatedData.content[event.index];
          if (block.type === 'tool_use' && typeof block.input === 'string') {
            try {
              block.input = JSON.parse(block.input as string);
            } catch (error) {
              throw PixiAgentErrorBuilder.modelResponseError(
                `Failed to parse input JSON for content block at index ${event.index}: ${(error as Error).message}`,
              );
            }
          }
          break;
        }
        case 'message_stop':
          break;
        default:
          assertNever(event);
      }
    }

    return this.toModelResponse(streamDataExtractor.accumulatedData);
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
    if (ErrorGuards.isPixiAgentError(error)) {
      return error;
    }
    const signal = requestOptions?.signal;
    if (signal?.aborted) {
      const reason = signal.reason;
      if (ErrorGuards.isPixiAgentError(reason)) {
        return reason;
      }
      if (typeof reason === 'string' && reason.length > 0) {
        return PixiAgentErrorBuilder.agentInterrupted(reason);
      }
      if (reason instanceof Error && reason.message) {
        return PixiAgentErrorBuilder.agentInterrupted(reason.message);
      }
      return PixiAgentErrorBuilder.agentInterrupted('Abort signal triggered');
    }
    if (ErrorGuards.isLikelyAbortError(error)) {
      return PixiAgentErrorBuilder.agentInterrupted((error as Error).message);
    }
    if (!ErrorGuards.isLikelyTimeoutError(error)) {
      return error;
    }
    return PixiAgentErrorBuilder.modelRequestTimeout(
      this.client.baseURL ?? AnthropicTransport.OFFICIAL_BASE_URL,
      requestOptions?.timeout,
      undefined,
      error,
    );
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
        input: part.arguments === '' ? null : JSON.parse(part.arguments),
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
        arguments: ContentParts.createProviderToolCallArguments('anthropic', block.type, block),
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

const BlockParamContentPartHelper = {
  toParts(
    block: ContentBlockParam | ContentBlock | ToolReferenceBlockParam,
  ): ContentPart | Array<ContentPart> {
    switch (block.type) {
      case 'text':
        return BlockParamContentPartHelper.toTextPart(block);
      case 'thinking':
        return BlockParamContentPartHelper.toThinkingPart(block);
      case 'image':
        return BlockParamContentPartHelper.toImagePart(block);
      case 'document':
        return BlockParamContentPartHelper.toDocumentPart(block);
      case 'tool_use':
      case 'server_tool_use':
        return BlockParamContentPartHelper.toToolCallPart(block);
      case 'tool_result':
        return BlockParamContentPartHelper.toToolResultPart(block);
      case 'container_upload': // ContainerUploadBlockParam | ContainerUploadBlock
        return BlockParamContentPartHelper.toContainerDocumentPart(block);
      case 'bash_code_execution_tool_result': // BashCodeExecutionToolResultBlockParam | BashCodeExecutionToolResultBlock
      case 'code_execution_tool_result': // CodeExecutionToolResultBlockParam | CodeExecutionToolResultBlock
      case 'text_editor_code_execution_tool_result': // TextEditorCodeExecutionToolResultBlockParam | TextEditorCodeExecutionToolResultBlock
      case 'tool_search_tool_result': // ToolSearchToolResultBlockParam | ToolSearchToolResultBlock
      case 'web_fetch_tool_result': // WebFetchToolResultBlockParam | WebFetchToolResultBlock
      case 'web_search_tool_result': // WebSearchToolResultBlockParam | WebSearchToolResultBlock
        return BlockParamContentPartHelper.toSpecificToolResultPart(block);
      case 'tool_reference': // server side result
        return BlockParamContentPartHelper.toToolReferenceTextPart(block);
      case 'search_result': // server side result SearchResultBlockParam | SearchResultBlock
        return BlockParamContentPartHelper.toSearchResultTextPart(block);
      case 'redacted_thinking':
        return [];
      default:
        assertNever(block);
    }
  },

  toTextPart(block: TextBlockParam | TextBlock): TextPart {
    return {
      type: 'text',
      text: block.text,
      citations: block.citations?.map(BlockParamContentPartHelper.toCitation),
    };
  },

  toCitation(citation: TextCitation | TextCitationParam): Citation {
    switch (citation.type) {
      case 'char_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_char_index,
          endIndex: citation.end_char_index,
          indexType: 'char',
          source: `document index: ${citation.document_index}`,
        };
      case 'page_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_page_number,
          endIndex: citation.end_page_number,
          indexType: 'page',
          source: `document index: ${citation.document_index}`,
        };
      case 'content_block_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_block_index,
          endIndex: citation.end_block_index,
          indexType: 'block',
          source: `document index: ${citation.document_index}`,
        };
      case 'search_result_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.title ?? undefined,
          startIndex: citation.start_block_index,
          endIndex: citation.end_block_index,
          indexType: 'block',
          source: citation.source,
        };
      case 'web_search_result_location':
        return {
          type: 'web_location',
          citedText: citation.cited_text,
          title: citation.title ?? undefined,
          url: citation.url,
        };
    }
  },

  toImagePart(block: ImageBlockParam): ImagePart {
    switch (block.source.type) {
      case 'base64':
        return {
          type: 'image',
          image: {
            sourceType: 'base64',
            mimeType: block.source.media_type,
            data: block.source.data,
          },
        };
      case 'url':
        return { type: 'image', image: { sourceType: 'url', url: block.source.url } };
    }
  },

  toDocumentPart(block: DocumentBlockParam): DocumentPart | Array<TextPart | ImagePart> {
    switch (block.source.type) {
      case 'base64':
        return {
          type: 'document',
          document: {
            sourceType: 'base64',
            mimeType: block.source.media_type,
            data: block.source.data,
          },
        };
      case 'text':
        return [
          {
            type: 'text',
            text: block.source.data,
          },
        ];
      case 'url':
        return { type: 'document', document: { sourceType: 'url', url: block.source.url } };
      case 'content': {
        if (typeof block.source.content === 'string') {
          return [
            {
              type: 'text',
              text: block.source.content,
            },
          ];
        }
        return block.source.content
          .map((b) => BlockParamContentPartHelper.toParts(b))
          .flat()
          .filter(
            (part): part is TextPart | ImagePart => part.type === 'text' || part.type === 'image',
          );
      }
    }
  },

  toThinkingPart(block: ThinkingBlockParam | ThinkingBlock): ThinkingPart {
    switch (block.type) {
      case 'thinking':
        return { type: 'thinking', content: block.thinking, signature: block.signature };
      // | RedactedThinkingBlockParam | RedactedThinkingBlock, the content in these blocks is
      // unreadable. It's useless to convert them to ThinkingPart.
    }
  },

  toToolCallPart(
    block: ToolUseBlockParam | ToolUseBlock | ServerToolUseBlockParam | ServerToolUseBlock,
  ): ToolCallPart | ServerToolUsePart {
    switch (block.type) {
      case 'tool_use': {
        if (!block.caller || block.caller.type === 'direct') {
          return {
            type: 'tool_call',
            id: block.id,
            name: block.name,
            arguments: JSON.stringify(block.input),
          };
        } else {
          const { name, ...rest } = block;
          return {
            type: 'server_tool_use',
            name: name,
            data: JSON.stringify(rest),
            providerSpecific: ApiModes.ANTHROPIC,
          };
        }
      }
      case 'server_tool_use': {
        if (block.caller && block.caller.type === 'direct') {
          const { id, name, ...rest } = block;
          return {
            type: 'tool_call',
            id: id,
            name: name,
            arguments: JSON.stringify(rest),
            providerSpecific: ApiModes.ANTHROPIC,
          };
        } else {
          const { name, ...rest } = block;
          return {
            type: 'server_tool_use',
            name: name,
            data: JSON.stringify(rest),
            providerSpecific: ApiModes.ANTHROPIC,
          };
        }
      }
      default:
        assertNever(block);
    }
  },

  toToolResultPart(block: ToolResultBlockParam): ToolResultPart {
    if (!block.content || typeof block.content === 'string') {
      return {
        type: 'tool_result',
        id: block.tool_use_id,
        result: block.content,
        isError: block.is_error ?? undefined,
      };
    }
    return {
      type: 'tool_result',
      id: block.tool_use_id,
      result: JSON.stringify(
        block.content
          .map((b) => {
            const parts = BlockParamContentPartHelper.toParts(b);
            if (Array.isArray(parts)) {
              return parts;
            }
            return [parts];
          })
          .flat(),
      ),
      isError: block.is_error ?? undefined,
    };
  },
  toContainerDocumentPart(block: ContainerUploadBlockParam | ContainerUploadBlock): DocumentPart {
    return {
      type: 'document',
      document: {
        sourceType: 'file_id',
        fileId: block.file_id,
      },
    };
  },
  toSpecificToolResultPart(
    block:
      | BashCodeExecutionToolResultBlockParam
      | BashCodeExecutionToolResultBlock
      | CodeExecutionToolResultBlockParam
      | CodeExecutionToolResultBlock
      | TextEditorCodeExecutionToolResultBlockParam
      | TextEditorCodeExecutionToolResultBlock
      | ToolSearchToolResultBlockParam
      | ToolSearchToolResultBlock
      | WebFetchToolResultBlockParam
      | WebFetchToolResultBlock
      | WebSearchToolResultBlockParam
      | WebSearchToolResultBlock,
  ): ToolResultPart {
    const { tool_use_id, type, ...rest } = block;

    return {
      type: 'tool_result',
      id: tool_use_id,
      name: type.replace('_tool_result', ''),
      result: JSON.stringify(rest),
      providerSpecific: ApiModes.ANTHROPIC,
    };
  },
  toToolReferenceTextPart(block: ToolReferenceBlockParam | ToolReferenceBlock): TextPart {
    return {
      type: 'text',
      text: JSON.stringify({
        tool_name: block.tool_name,
        type: block.type,
      }),
    };
  },
  toSearchResultTextPart(block: SearchResultBlockParam): TextPart {
    return {
      type: 'text',
      text: JSON.stringify({
        source: block.source,
        title: block.title,
        content: block.content,
        type: block.type,
      }),
    };
  },
};

const ContentPartBlockParamHelper = {
  toBlockParam(part: ContentPart): ContentBlockParam | null {
    switch (part.type) {
      case 'text':
      case 'refusal':
        return ContentPartBlockParamHelper.toTextBlockParam(part);
      case 'thinking':
        return ContentPartBlockParamHelper.toThinkingBlockParam(part);
      case 'image':
        return ContentPartBlockParamHelper.toImageBlockParam(part);
      case 'document':
        return ContentPartBlockParamHelper.toDocumentBlockParam(part);
      case 'tool_call':
        return ContentPartBlockParamHelper.toToolUseBlockParam(part);
      case 'tool_result':
        return ContentPartBlockParamHelper.toToolResultBlockParam(part);
      case 'server_tool_use':
        return ContentPartBlockParamHelper.toServerToolUseBlockParam(part);
      case 'audio':
        return ContentPartBlockParamHelper.toAudioTextBlockParam(part);
      case 'video':
        return ContentPartBlockParamHelper.toVideoTextBlockParam(part);
      default:
        assertNever(part);
    }
  },

  toTextBlockParam(part: TextPart | RefusalPart): TextBlockParam {
    return {
      type: 'text',
      text: 'text' in part ? part.text : part.reason,
      citations:
        'citations' in part && part.citations
          ? part.citations
              .map((citation) => ContentPartBlockParamHelper.toCitationParam(citation))
              .filter((citation): citation is TextCitationParam => citation !== null)
          : undefined,
    };
  },
  toCitationParam(citation: Citation): TextCitationParam | null {
    switch (citation.type) {
      case 'file_location':
      case 'others_location':
        return null;
      case 'web_location':
        return {
          type: 'web_search_result_location',
          cited_text: citation.citedText,
          title: citation.title ?? null,
          url: citation.url,
          encrypted_index: '',
        };
    }
  },
  toThinkingBlockParam(part: ThinkingPart): ThinkingBlockParam {
    return {
      type: 'thinking',
      thinking: part.content,
      signature: part.signature ?? '',
    };
  },
  toImageBlockParam(part: ImagePart): ImageBlockParam | null {
    switch (part.image.sourceType) {
      case 'base64':
        switch (part.image.mimeType) {
          case 'image/jpeg':
          case 'image/png':
          case 'image/gif':
          case 'image/webp':
            return {
              type: 'image',
              source: {
                type: 'base64',
                media_type: part.image.mimeType,
                data: part.image.data,
              },
            };
          default:
            return null;
        }
      case 'url':
        return {
          type: 'image',
          source: { type: 'url', url: part.image.url },
        };
      case 'file_id':
        return null;
    }
  },
  toDocumentBlockParam(part: DocumentPart): DocumentBlockParam | null {
    switch (part.document.sourceType) {
      case 'base64':
        switch (part.document.mimeType) {
          case 'text/plain':
            return {
              type: 'document',
              source: {
                type: 'text',
                media_type: 'text/plain',
                data: part.document.data!,
              },
            };
          case 'application/pdf':
            return {
              type: 'document',
              source: {
                type: 'base64',
                media_type: 'application/pdf',
                data: part.document.data!,
              },
            };
          default:
            return null;
        }
      case 'url':
        return {
          type: 'document',
          source: { type: 'url', url: part.document.url! },
        };
      case 'file_id':
        return null;
    }
  },
  toToolUseBlockParam(part: ToolCallPart): ToolUseBlockParam | ServerToolUseBlockParam {
    if (part.providerSpecific && part.providerSpecific === ApiModes.ANTHROPIC) {
      return {
        ...JSON.parse(part.arguments),
        id: part.id,
        name: part.name,
      };
    }
    return {
      type: 'tool_use',
      id: part.id,
      name: part.name,
      input: part.arguments === '' ? null : JSON.parse(part.arguments),
    };
  },
  toToolResultBlockParam(
    part: ToolResultPart,
  ):
    | ToolResultBlockParam
    | BashCodeExecutionToolResultBlockParam
    | BashCodeExecutionToolResultBlock
    | CodeExecutionToolResultBlockParam
    | CodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlockParam
    | TextEditorCodeExecutionToolResultBlock
    | ToolSearchToolResultBlockParam
    | ToolSearchToolResultBlock
    | WebFetchToolResultBlockParam
    | WebFetchToolResultBlock
    | WebSearchToolResultBlockParam
    | WebSearchToolResultBlock {
    try {
      const parsedResult = !part.result || part.result === '' ? null : JSON.parse(part.result);

      if (part.providerSpecific === ApiModes.ANTHROPIC) {
        switch (part.name) {
          case 'bash_code_execution': // BashCodeExecutionToolResultBlockParam | BashCodeExecutionToolResultBlock
          case 'code_execution': // CodeExecutionToolResultBlockParam | CodeExecutionToolResultBlock
          case 'text_editor_code_execution': // TextEditorCodeExecutionToolResultBlockParam | TextEditorCodeExecutionToolResultBlock
          case 'tool_search': // ToolSearchToolResultBlockParam | ToolSearchToolResultBlock
          case 'web_fetch': // WebFetchToolResultBlockParam | WebFetchToolResultBlock
          case 'web_search': {
            // WebSearchToolResultBlockParam | WebSearchToolResultBlock
            const resultObj =
              typeof parsedResult === 'object' && parsedResult !== null ? parsedResult : {};
            return {
              ...resultObj,
              type: `${part.name}_tool_result`,
              tool_use_id: part.id,
            };
          }
        }
      }

      if (parsedResult && Array.isArray(parsedResult)) {
        const convertable = parsedResult.every((item) => {
          if (!('type' in item)) return false;
          switch (item.type) {
            case 'text':
              return 'text' in item;
            case 'image':
              return (
                item.image &&
                ((item.image.sourceType === 'base64' && 'data' in item.image) ||
                  (item.image.sourceType === 'url' && 'url' in item.image))
              );
            case 'document':
              return (
                item.document &&
                ((item.document.sourceType === 'base64' && 'data' in item.document) ||
                  (item.document.sourceType === 'url' && 'url' in item.document) ||
                  (item.document.sourceType === 'text' && 'data' in item.document))
              );
            default:
              return false;
          }
        });
        if (convertable) {
          return {
            type: 'tool_result',
            tool_use_id: part.id,
            content: parsedResult.map(
              (item) =>
                ContentPartBlockParamHelper.toBlockParam(item) as
                  | TextBlockParam
                  | ImageBlockParam
                  | DocumentBlockParam
                  | SearchResultBlockParam
                  | ToolReferenceBlockParam,
            ),
            is_error: part.isError ?? undefined,
          };
        }
      }
    } catch {
      // empty
    }
    return {
      type: 'tool_result',
      tool_use_id: part.id,
      content: part.result,
      is_error: part.isError ?? undefined,
    };
  },
  toServerToolUseBlockParam(
    part: ServerToolUsePart,
  ): ToolUseBlockParam | ServerToolUseBlockParam | TextBlockParam {
    if (part.providerSpecific === ApiModes.ANTHROPIC) {
      try {
        return {
          ...JSON.parse(part.data ?? '{}'),
          name: part.name,
        };
      } catch {
        // empty
      }
    }
    return {
      type: 'text',
      text: `Tool use: ${part.name} with data ${part.data}`,
    };
  },
  toAudioTextBlockParam(part: AudioPart): TextBlockParam {
    return {
      type: 'text',
      text: JSON.stringify({
        audio: part.audio,
        type: part.type,
      }),
    };
  },
  toVideoTextBlockParam(part: VideoPart): TextBlockParam {
    return {
      type: 'text',
      text: JSON.stringify({
        video: part.video,
        type: part.type,
      }),
    };
  },
};

const ConvertHelper = {
  toBlockParam: ContentPartBlockParamHelper.toBlockParam,
  toParts: BlockParamContentPartHelper.toParts,
};
