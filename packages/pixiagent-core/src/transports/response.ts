import { OpenAI } from 'openai/client';
import type {
  Response,
  ResponseCreateParamsStreaming,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseReasoningItem,
  ToolChoiceOptions,
  ResponseInputContent,
  ResponseOutputText,
  ResponseOutputRefusal,
  ResponseInputText,
  ResponseInputImage,
  ResponseInputFile,
  ResponseInputImageContent,
  ResponseInputFileContent,
  ResponseComputerToolCall,
  ResponseCustomToolCall,
  ResponseToolSearchCall,
  ResponseToolSearchOutputItemParam,
  ResponseToolSearchOutputItem,
  ResponseCustomToolCallOutputItem,
  ResponseApplyPatchToolCallOutput,
  ResponseFunctionShellToolCallOutput,
  ResponseCustomToolCallOutput,
  ResponseComputerToolCallOutputItem,
  ResponseApplyPatchToolCall,
  ResponseFunctionShellToolCall,
  EasyInputMessage,
  ResponseErrorEvent,
} from 'openai/resources/responses/responses';
import {
  ApiModes,
  Citation,
  ContentPart,
  DocumentPart,
  ImagePart,
  RawDeltaMessageType,
  RawLLMParametersType,
  RawResponseType,
  RefusalPart,
  RoleType,
  SessionMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ResponseApiMessage,
  ThinkingPart,
  ModelStopReasons,
} from '../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../errors';
import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ProviderTransport,
  StreamCallbacks,
  StreamDataExtractor,
} from './base';
import { randomUUID } from 'crypto';

export class ResponseTransport extends ProviderTransport<ResponseApiMessage> {
  readonly client: OpenAI;
  private static readonly OFFICIAL_BASE_URL = 'https://api.openai.com/v1';
  private readonly configuredBaseUrl?: string;

  private static normalizeBaseUrl(baseUrl?: string): string | undefined {
    if (!baseUrl) return baseUrl;
    const normalized = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    // Some docs provide the full responses endpoint; SDK baseURL should be the parent API root.
    if (normalized.toLowerCase().endsWith('/responses')) {
      return normalized.slice(0, -'/responses'.length);
    }
    return normalized;
  }

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      ResponseApiMessage,
      RawDeltaMessageType,
      RawLLMParametersType,
      RawResponseType
    >,
  ) {
    super(ApiModes.RESPONSE, dialectResolver);
    this.configuredBaseUrl = ResponseTransport.normalizeBaseUrl(baseUrl);
    this.client = new OpenAI({
      baseURL: this.configuredBaseUrl,
      apiKey,
    });
  }

  convertFromRawMessage(rawMsg: ResponseApiMessage): SessionMessage {
    const parts = rawMsg.content.flatMap(ConvertHelper.toContentParts);
    const message: SessionMessage = {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: rawMsg.role,
      content: parts,
      modelResponseInfo: rawMsg.modelResponseInfo,
      metadata: rawMsg.metadata,
    };
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(message, rawMsg) : message;
  }

  convertToRawMessage(msg: SessionMessage): ResponseApiMessage {
    const items = ConvertHelper.toResponseItems(msg.role, msg.content);
    const responseApiMessage: ResponseApiMessage = {
      messageId: msg.messageId,
      type: 'response_api_message',
      role: msg.role,
      content: items,
      modelResponseInfo: msg.modelResponseInfo,
      metadata: msg.metadata,
    };

    return this.dialectResolver
      ? this.dialectResolver.manipulateRawMessage(responseApiMessage, msg)
      : responseApiMessage;
  }

  private getToolChoice(
    options: ModelOptions,
  ): ToolChoiceOptions | 'required' | 'none' | undefined {
    if (!options.tools || options.tools.length === 0) {
      return undefined;
    }

    switch (options.toolChoice) {
      case 'force':
        return 'required';
      case 'none':
        return 'none';
      default:
        return 'auto';
    }
  }

  private getReasoningEffort(
    thinkEffort?: ModelOptions['thinkEffort'],
  ): 'none' | 'low' | 'medium' | 'high' | 'xhigh' | undefined {
    switch (thinkEffort) {
      case 'disable':
        return 'none';
      case 'low':
        return 'low';
      case 'medium':
        return 'medium';
      case 'high':
        return 'high';
      case 'extreme':
        return 'xhigh';
      default:
        return undefined;
    }
  }

  private getStopReason(response: Response): ModelStopReasons {
    const hasToolCall = response.output.some(
      (item) =>
        item.type === 'function_call' ||
        item.type === 'computer_call' ||
        item.type === 'file_search_call' ||
        item.type === 'custom_tool_call' ||
        item.type === 'tool_search_call' ||
        item.type === 'web_search_call' ||
        item.type === 'code_interpreter_call' ||
        item.type === 'image_generation_call' ||
        item.type === 'local_shell_call' ||
        item.type === 'shell_call' ||
        item.type === 'apply_patch_call' ||
        item.type === 'mcp_call' ||
        item.type === 'mcp_approval_request',
    );
    if (hasToolCall) {
      return ModelStopReasons.TOOL_CALL;
    }

    if (response.status === 'cancelled') {
      return ModelStopReasons.CANCELLED;
    }

    if (response.status === 'incomplete') {
      switch (response.incomplete_details?.reason) {
        case 'max_output_tokens':
          return ModelStopReasons.MAX_TOKENS;
        case 'content_filter':
          return ModelStopReasons.REFUSAL;
        default:
          return ModelStopReasons.MAX_TOKENS;
      }
    }
    if (response.status === 'failed') {
      return ModelStopReasons.CANCELLED;
    }

    return ModelStopReasons.OTHERS;
  }

  private getResponseMessage(response: Response): Omit<ResponseApiMessage, 'messageId'> {
    const supportedOutputItems: ResponseInputItem[] = [];
    for (const item of response.output) {
      if (item.type === 'function_call' || item.type === 'message' || item.type === 'reasoning') {
        supportedOutputItems.push(item as ResponseInputItem);
      }
    }

    const metadata: Record<string, unknown> = {
      pixiagent_response_id: response.id,
      pixiagent_response_error: response.error,
      pixiagent_response_incomplete_details: response.incomplete_details,
      pixiagent_response_status: response.status,
    };

    return {
      type: 'response_api_message',
      role: 'assistant',
      content: response.output,
      modelResponseInfo: {
        responseId: response.id,
        responseModel: response.model,
        stopReason: this.getStopReason(response),
        refusal: this.getRefusal(response.output),
        usage: response.usage
          ? {
              inputTokens: response.usage.input_tokens,
              outputTokens: response.usage.output_tokens,
              totalTokens: response.usage.total_tokens,
              reasoningTokens: response.usage.output_tokens_details?.reasoning_tokens,
              cacheReadTokens: response.usage.input_tokens_details?.cached_tokens,
              inputTokenDetails: response.usage.input_tokens_details
                ? { ...response.usage.input_tokens_details }
                : undefined,
              outputTokenDetails: response.usage.output_tokens_details
                ? { ...response.usage.output_tokens_details }
                : undefined,
            }
          : undefined,
      },
      metadata,
    };
  }

  private getRefusal(content: Array<ResponseInputItem | ResponseOutputItem>): string | undefined {
    for (const item of content) {
      if (item.type === 'message' && item.role === 'assistant' && Array.isArray(item.content)) {
        const refusal = item.content.find((content) => content.type === 'refusal');
        if (refusal?.type === 'refusal') {
          return refusal.refusal;
        }
      }
    }
    return undefined;
  }

  private buildParams(
    options: ModelOptions,
    messages: Array<ResponseApiMessage>,
  ): ResponseCreateParamsStreaming {
    const params: ResponseCreateParamsStreaming = {
      model: options.model,
      input: messages.flatMap((message) =>
        message.content.map((item) => {
          if ('id' in item) {
            const inputItem = { ...item };
            delete inputItem.id;
            return inputItem as ResponseInputItem;
          }
          return item as ResponseInputItem;
        }),
      ),
      instructions: options.systemPrompt,
      max_output_tokens: options.maxTokens,
      temperature: options.temperature,
      metadata: options.metadata,
      parallel_tool_calls: options.parallelToolCalls,
      tool_choice: this.getToolChoice(options),
      store: false,
      stream: true,
      reasoning: options.thinkEffort
        ? {
            effort: this.getReasoningEffort(options.thinkEffort),
            summary: 'auto',
          }
        : undefined,
      text: options.outputSchema
        ? {
            format: {
              type: 'json_schema',
              name: 'structured_output',
              schema: options.outputSchema.schema,
              strict: options.outputSchema.strict,
            },
          }
        : undefined,
      tools: options.tools?.map((tool) => ({
        type: 'function' as const,
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
        strict: true,
      })),
    };

    return params;
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

  async generate(
    options: ModelOptions,
    messages: Array<ResponseApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<ResponseApiMessage, 'messageId'>> {
    const params = this.buildParams(options, messages);

    try {
      const stream = await this.client.responses.create(
        { ...params, stream: true },
        this.getStreamRequestOptions(requestOptions),
      );

      const streamDataExtractor = new StreamDataExtractor(
        {
          content: Array<ResponseInputItem | ResponseOutputItem>(),
          response: undefined as Response | undefined,
        },
        callbacks,
      );

      for await (const event of stream) {
        switch (event.type) {
          case 'response.created':
          case 'response.queued':
          case 'response.in_progress':
            break;
          case 'error':
            throw this.extractResponseError(event as ResponseErrorEvent);
          case 'response.completed':
          case 'response.incomplete':
          case 'response.failed': {
            streamDataExtractor.accumulatedData.response = event.response;
            break;
          }
          case 'response.output_item.added': {
            streamDataExtractor.accumulate(
              { key: `output_item_${event.output_index}`, value: event.item },
              (accumulated, data) => {
                if (accumulated.content.length !== event.output_index) {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received output item for non-existing index ${event.output_index}`,
                    this.client.baseURL,
                    'invalid_stream_event',
                    { event },
                  );
                }
                accumulated.content.push(data);
              },
              (_existing, _newData) => {
                // should be always new item in this event.
              },
              (delta) => {
                switch (delta.type) {
                  case 'function_call':
                    return ConvertHelper.toContentPart(delta);
                  default:
                    return null;
                }
              },
            );
            break;
          }
          case 'response.output_item.done': {
            streamDataExtractor.accumulate(
              { key: `output_item_${event.output_index}`, value: event.item },
              (_accumulated, _data) => {},
              (existing, newData) => {
                if (streamDataExtractor.accumulatedData.content.length <= event.output_index) {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received output item done event for non-existing index ${event.output_index}`,
                    this.client.baseURL,
                    'invalid_stream_event',
                    { event },
                  );
                }
                const item = streamDataExtractor.accumulatedData.content[event.output_index];
                if (existing !== item) {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Data mismatch for output item at index ${event.output_index} between added and done events`,
                    this.client.baseURL,
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
                    return ConvertHelper.toContentPart(
                      delta as Exclude<
                        ResponseOutputItem,
                        ResponseOutputMessage | ResponseFunctionToolCall
                      >,
                    );
                }
              },
            );

            break;
          }
          case 'response.content_part.added': {
            // set item to the message.content
            streamDataExtractor.accumulate(
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
                    this.client.baseURL,
                    'invalid_stream_event',
                    { event },
                  );
                }
                switch (content.type) {
                  case 'message': {
                    if (data.type !== 'output_text' && data.type !== 'refusal') {
                      throw PixiAgentErrorBuilder.modelResponseError(
                        `Expected content part of type output_text or refusal for message item, but got ${data.type}`,
                        this.client.baseURL,
                        'invalid_stream_event',
                        { event },
                      );
                    }
                    const message = accumulated.content[
                      event.output_index
                    ] as ResponseOutputMessage;
                    if (message.content.length !== event.content_index) {
                      throw PixiAgentErrorBuilder.modelResponseError(
                        `Received out-of-order content part with content_index ${event.content_index} for message item, expected content_index ${message.content.length}`,
                        this.client.baseURL,
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
                        this.client.baseURL,
                        'invalid_stream_event',
                        { event },
                      );
                    }
                    const reasoningItem = accumulated.content[
                      event.output_index
                    ] as ResponseReasoningItem;
                    if (!reasoningItem.content) {
                      reasoningItem.content = [];
                    }
                    reasoningItem.content.push(data);
                    break;
                  }
                  default: {
                    throw PixiAgentErrorBuilder.modelResponseError(
                      `Received content part for unsupported item type ${content.type}`,
                      this.client.baseURL,
                      'invalid_stream_event',
                      { event },
                    );
                  }
                }
              },
              (_existing, _newData) => {
                // no data to merge in this event
              },
              (delta) => {
                switch (delta.type) {
                  case 'output_text': {
                    return delta.text === '' ? null : { type: 'text', text: delta.text };
                  }
                  case 'refusal':
                    return delta.refusal === '' ? null : { type: 'refusal', reason: delta.refusal };
                  case 'reasoning_text':
                    return delta.text === '' ? null : { type: 'thinking', content: delta.text };
                }
              },
            );
            break;
          }
          case 'response.content_part.done':
            break;
          case 'response.output_text.delta': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_content_${event.content_index}`,
                value: {
                  type: 'output_text',
                  text: event.delta,
                } as ResponseOutputText,
              },
              (_accumulated, _data) => {}, // no data to append
              (existing, newData) => {
                existing.text += newData.text;
              },
              (delta) => delta.text === '' ? null : { type: 'text', text: delta.text },
            );
            break;
          }
          case 'response.output_text.annotation.added': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_content_${event.content_index}`,
                value: {
                  type: 'output_text',
                  text: '',
                  annotations: [event.annotation],
                } as ResponseOutputText,
              },
              (_accumulated, _data) => {}, // no data to append
              (existing, newData) => {
                if (!existing.annotations) {
                  existing.annotations = [];
                }
                existing.annotations.push(...newData.annotations);
              },
              (delta) => ConvertHelper.toContentPart(delta),
            );
            break;
          }
          case 'response.output_text.done':
            // no need to do anything as the final text should have been accumulated in the delta events, but we can use this event to trigger any callbacks if needed.
            break;

          case 'response.refusal.delta': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_content_${event.content_index}`,
                value: {
                  type: 'refusal',
                  refusal: event.delta,
                } as ResponseOutputRefusal,
              },
              (_accumulated, _data) => {}, // no data to append
              (existing, newData) => {
                existing.refusal += newData.refusal;
              },
              (delta) => delta.refusal === '' ? null : ConvertHelper.toContentPart(delta),
            );
            break;
          }
          case 'response.refusal.done':
            break;
          case 'response.reasoning_text.delta': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_content_${event.content_index}`,
                value: {
                  type: 'reasoning_text',
                  text: event.delta,
                },
              },
              (_accumulated, _data) => {}, // Item is added `response.content_part.added`
              (existing, newData) => {
                existing.text += newData.text;
              },
              (delta) => delta.text === '' ? null : { type: 'thinking', content: delta.text },
            );
            break;
          }
          case 'response.reasoning_text.done':
            break;
          case 'response.reasoning_summary_part.added': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_summary_${event.summary_index}`,
                value: event.part,
              },
              (accumulated, data) => {
                if (accumulated.content.length <= event.output_index) {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received reasoning summary part for non-existing output item at index ${event.output_index}`,
                    this.client.baseURL,
                    'invalid_stream_event',
                    { event },
                  );
                }
                const content = accumulated.content[event.output_index];
                if (content.type !== 'reasoning') {
                  throw PixiAgentErrorBuilder.modelResponseError(
                    `Received reasoning summary part for output item at index ${event.output_index} which is not a reasoning item`,
                    this.client.baseURL,
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
                    this.client.baseURL,
                    'invalid_stream_event',
                    { event },
                  );
                }
                content.summary.push(data);
              },
              (_existing, _newData) => {}, // no data to merge in this event
              (delta) => {
                if (delta.text === '') {
                  return null;
                }
                return { type: 'thinking', content: delta.text };
              },
            );
            break;
          }
          case 'response.reasoning_summary_text.delta': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}_summary_${event.summary_index}`,
                value: {
                  type: 'summary_text',
                  text: event.delta,
                },
              },
              (_accumulated, _data) => {}, // summary item is added in `response.reasoning_summary_part.added`
              (existing, newData) => {
                if (!existing.text) {
                  existing.text = '';
                }
                existing.text += newData.text;
              },
              (delta) => delta.text === '' ? null : { type: 'thinking', content: delta.text },
            );
            break;
          }
          case 'response.reasoning_summary_text.done':
          case 'response.reasoning_summary_part.done':
            break;
          case 'response.function_call_arguments.delta': {
            streamDataExtractor.accumulate(
              {
                key: `output_item_${event.output_index}`,
                value: {
                  arguments: event.delta,
                  call_id: '',
                  name: '',
                  type: 'function_call',
                } as ResponseFunctionToolCall,
              },
              (_accumulated, _data) => {},
              (existing, newData) => {
                existing.arguments += newData.arguments;
                newData.call_id = existing.call_id;
                newData.name = existing.name;
              },
              (delta) => delta.arguments === '' ? null : ConvertHelper.toContentPart(delta),
            );
            break;
          }
          case 'response.function_call_arguments.done':
            break;
          case 'response.audio.delta':
          case 'response.audio.done':
          case 'response.audio.transcript.delta':
          case 'response.audio.transcript.done':
          case 'response.code_interpreter_call.completed':
          case 'response.image_generation_call.completed':
          case 'response.code_interpreter_call.in_progress':
          case 'response.code_interpreter_call.interpreting':
          case 'response.code_interpreter_call_code.delta':
          case 'response.code_interpreter_call_code.done':
          case 'response.image_generation_call.in_progress':
          case 'response.custom_tool_call_input.delta':
          case 'response.custom_tool_call_input.done':
          case 'response.file_search_call.completed':
          case 'response.file_search_call.in_progress':
          case 'response.web_search_call.completed':
          case 'response.file_search_call.searching':
          case 'response.image_generation_call.generating':
          case 'response.image_generation_call.partial_image':
          case 'response.mcp_call.completed':
          case 'response.mcp_call.failed':
          case 'response.mcp_call.in_progress':
          case 'response.mcp_call_arguments.delta':
          case 'response.mcp_call_arguments.done':
          case 'response.mcp_list_tools.completed':
          case 'response.mcp_list_tools.failed':
          case 'response.mcp_list_tools.in_progress':
          case 'response.web_search_call.in_progress':
          case 'response.web_search_call.searching':
            break;
        }
      }
      const response = streamDataExtractor.accumulatedData.response;
      if (!response) {
        throw PixiAgentErrorBuilder.modelResponseError(
          'Response stream ended without a terminal response event',
          this.client.baseURL,
          'invalid_stream_event',
        );
      }

      return this.getResponseMessage(response);
    } catch (error) {
      throw this.wrapRequestError(error, requestOptions);
    }
  }

  private extractResponseError(event: ResponseErrorEvent): Error {
    switch (event.code) {
      case 'rate_limit_exceeded':
      case 'vector_store_timeout':
        return PixiAgentErrorBuilder.modelRequestRetriableError(
          event.message ?? 'Request failed with retriable error',
          this.client.baseURL,
          event.code,
        );
      case 'server_error':
        return PixiAgentErrorBuilder.modelResponseError(
          event.message ?? 'Server error during model response',
          this.client.baseURL,
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
      this.client.baseURL ?? ResponseTransport.OFFICIAL_BASE_URL,
      requestOptions?.timeout,
      undefined,
      error,
    );
  }
}

const ResponseContentPartHelper = {
  /**
   * Map input and output response item variants into shared ContentPart mappings.
   *
   * Equivalence analysis:
   * - message / output message share the same content mapping behavior.
   * - function_call is a shared tool-invocation form.
   * - reasoning is a shared thinking/result summary form.
   * - image_generation_call appears on both sides as the same underlying image generation event.
   * - tool call families such as computer_call, file_search_call, custom_tool_call,
   *   tool_search_call, web_search_call, code_interpreter_call, local_shell_call,
   *   shell_call, apply_patch_call, mcp_call, mcp_list_tools, mcp_approval_request,
   *   and mcp_approval_response are conceptually generic tool calls.
   */
  toContentPartsFromInputItem(item: ResponseInputItem): ContentPart[] {
    return this.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromOutputItem(item: ResponseOutputItem): ContentPart[] {
    return this.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromResponseItem(item: ResponseInputItem | ResponseOutputItem): ContentPart[] {
    const isResponseMessageItem = (
      data: ResponseInputItem | ResponseOutputItem,
    ): data is EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage => {
      return (
        data.type === 'message' ||
        (!data.type &&
          'content' in data &&
          (typeof data.content === 'string' ||
            (Array.isArray(data.content) &&
              data.content.every(
                (c) =>
                  c.type === 'input_text' || c.type === 'input_image' || c.type === 'input_file',
              ))))
      );
    };

    if (isResponseMessageItem(item)) {
      return this.fromResponseMessageItem(item);
    } else if (
      item.type === 'compaction' ||
      item.type === 'compaction_trigger' ||
      item.type === 'item_reference'
    ) {
      // These are control or metadata-only items and do not represent renderable
      // content that can be mapped to a ContentPart.
      return [];
    } else {
      const contentPart = this.fromResponseObject(item);
      if (contentPart) {
        return [contentPart];
      }
      return [];
    }
  },

  fromResponseMessageItem(
    item: EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage,
  ): ContentPart[] {
    if (typeof item.content === 'string') {
      return [
        {
          type: 'text',
          text: item.content,
        } as TextPart,
      ];
    }

    const contents: Array<
      | ResponseInputContent // ResponseInputItem.Message, `ResponseInputText | ResponseInputImage | ResponseInputFile`
      | ResponseOutputText // ResponseOutputMessage
      | ResponseOutputRefusal // ResponseOutputMessage
    > = item.content;

    return contents.map((c) => this.fromResponseObject(c)).filter((part) => part !== null);
  },

  fromResponseObject(
    item:
      | Exclude<
          ResponseInputItem,
          EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage
        >
      | Exclude<ResponseOutputItem, ResponseOutputMessage>
      | ResponseOutputText
      | ResponseInputText
      | ResponseOutputRefusal
      | ResponseInputImage
      | ResponseInputImageContent
      | ResponseInputFile
      | ResponseInputFileContent,
  ): ContentPart | null {
    switch (item.type) {
      case 'input_text':
        return this.fromResponseInputText(item);
      case 'output_text':
        return this.fromResponseOutputText(item);
      case 'refusal':
        return this.fromResponseOutputRefusal(item);
      case 'input_image':
        return this.fromResponseInputImage(item);
      case 'input_file':
        return this.fromResponseInputFile(item);
      case 'image_generation_call':
        return this.fromResponseImageGenerationCall(item);
      case 'function_call':
        return this.fromResponseFunctionCall(item);
      case 'function_call_output':
        return this.fromResponseFunctionCallOutput(item);
      case 'reasoning':
        return this.fromResponseReasoningItem(item);

      case 'computer_call':
      case 'custom_tool_call':
      case 'tool_search_call':
      case 'local_shell_call':
      case 'shell_call':
      case 'apply_patch_call':
        return this.fromResponseSpecificToolCall(item);
      case 'computer_call_output':
      case 'custom_tool_call_output':
      case 'tool_search_output':
      case 'local_shell_call_output':
      case 'shell_call_output':
      case 'apply_patch_call_output':
        return this.fromResponseSpecificToolOutput(item);
      // server tool calls (which includes the request argument and the response output)
      case 'file_search_call': // ResponseFileSearchToolCall
      case 'web_search_call': // ResponseFunctionWebSearch
      case 'code_interpreter_call': // ResponseCodeInterpreterToolCall
      case 'mcp_call': // ResponseInputItem.McpCall | ResponseOutputItem.McpCall
      case 'mcp_list_tools': // ResponseInputItem.McpListTools | ResponseOutputItem.McpListTools
        return null; // todo: consider a generic server side tool call part
      case 'mcp_approval_request': // ResponseInputItem.McpApprovalRequest | ResponseOutputItem.McpApprovalRequest
      case 'mcp_approval_response': // ResponseInputItem.McpApprovalResponse | ResponseOutputItem.McpApprovalResponse
      case 'compaction': // ResponseCompactionItemParam | ResponseCompactionItem
      case 'compaction_trigger': // ResponseInputItem.CompactionTrigger
      case 'item_reference': // ResponseInputItem.ItemReference | ResponseOutputItem.ItemReference
        return null; // todo: how to handle these types?
      case null: // ResponseInputItem.ItemReference
        return null;
      case undefined: // ResponseInputItem.ItemReference
        return null;
      default:
        return assertNever(item);
    }
  },

  fromResponseInputText(item: ResponseInputText): TextPart {
    return {
      type: 'text',
      text: item.text,
    };
  },

  fromResponseInputImage(item: ResponseInputImage | ResponseInputImageContent): ImagePart | null {
    if (item.image_url) {
      return {
        type: 'image',
        image: {
          sourceType: 'url',
          url: item.image_url,
        },
      } as ImagePart;
    } else if (item.file_id) {
      return {
        type: 'image',
        image: {
          sourceType: 'file_id',
          fileId: item.file_id,
        },
      } as ImagePart;
    }
    return null;
  },

  fromResponseInputFile(item: ResponseInputFile | ResponseInputFileContent): DocumentPart | null {
    if (item.file_id) {
      return {
        type: 'document',
        document: {
          fileId: item.file_id,
          fileName: item.filename,
        },
      } as DocumentPart;
    } else if (item.file_url) {
      return {
        type: 'document',
        document: {
          url: item.file_url,
          fileName: item.filename,
        },
      } as DocumentPart;
    } else if (item.file_data) {
      return {
        type: 'document',
        document: {
          data: item.file_data,
          mimeType: 'text/plain', // todo: infer the mime type from the filename extension or content if possible
          fileName: item.filename,
        },
      } as DocumentPart;
    }
    return null;
  },

  fromResponseOutputText(item: ResponseOutputText): TextPart {
    return {
      type: 'text',
      text: item.text,
      citations: item.annotations?.flatMap(this.fromAnnotation) ?? [],
    } as TextPart;
  },

  fromResponseOutputRefusal(item: ResponseOutputRefusal): RefusalPart {
    return {
      type: 'refusal',
      reason: item.refusal,
    };
  },

  fromResponseImageGenerationCall(
    item: ResponseOutputItem.ImageGenerationCall | ResponseInputItem.ImageGenerationCall,
  ): ImagePart | null {
    if (!item.result) {
      return null;
    }

    // The result is base64-encoded image data; mimeType is not guaranteed by the type.
    // Use a generic fallback and leave future refinement to later.
    return {
      type: 'image',
      image: {
        sourceType: 'base64',
        mimeType: 'image/png',
        data: item.result,
      },
    };
  },

  fromResponseFunctionCallOutput(item: ResponseInputItem.FunctionCallOutput): ToolResultPart {
    if (typeof item.output === 'string') {
      return {
        type: 'tool_result',
        id: item.call_id,
        result: item.output,
      };
    }

    return {
      type: 'tool_result',
      id: item.call_id,
      result: JSON.stringify(
        item.output.map((c) => this.fromResponseObject(c)).filter((part) => part !== undefined),
      ),
    };
  },

  fromResponseFunctionCall(item: ResponseFunctionToolCall): ToolCallPart {
    return {
      type: 'tool_call',
      id: item.call_id,
      name: item.name,
      arguments: item.arguments,
    };
  },

  fromResponseReasoningItem(item: ResponseReasoningItem): ThinkingPart | null {
    const summary = Array.isArray(item.summary)
      ? item.summary.map((summaryItem) => summaryItem.text).join('')
      : '';
    if (!summary) {
      return null;
    }

    return {
      type: 'thinking',
      content: summary,
    };
  },

  fromResponseSpecificToolCall(
    item:
      | ResponseComputerToolCall
      | ResponseCustomToolCall
      | ResponseInputItem.ToolSearchCall
      | ResponseToolSearchCall
      | ResponseInputItem.LocalShellCall
      | ResponseOutputItem.LocalShellCall
      | ResponseInputItem.ShellCall
      | ResponseFunctionShellToolCall
      | ResponseInputItem.ApplyPatchCall
      | ResponseApplyPatchToolCall,
  ): ToolCallPart {
    switch (item.type) {
      case 'computer_call':
      case 'custom_tool_call':
      case 'tool_search_call':
      case 'local_shell_call':
      case 'shell_call':
      case 'apply_patch_call':
        return {
          type: 'tool_call',
          id: item.call_id ?? 'tool_search_call_id', // the tool search call id may not be provided
          name: item.type,
          arguments: JSON.stringify(item),
          providerSpecific: ApiModes.RESPONSE,
        };
    }
  },

  fromResponseSpecificToolOutput(
    item:
      | ResponseInputItem.ComputerCallOutput
      | ResponseComputerToolCallOutputItem
      | ResponseCustomToolCallOutput
      | ResponseCustomToolCallOutputItem
      | ResponseToolSearchOutputItemParam
      | ResponseToolSearchOutputItem
      | ResponseInputItem.LocalShellCallOutput
      | ResponseOutputItem.LocalShellCallOutput
      | ResponseInputItem.ShellCallOutput
      | ResponseFunctionShellToolCallOutput
      | ResponseInputItem.ApplyPatchCallOutput
      | ResponseApplyPatchToolCallOutput,
  ): ToolResultPart {
    return {
      type: 'tool_result',
      // the tool call id may not be provided
      // the local shell output doesn't contain the call_id, but the LocalShellCall has.
      id: 'call_id' in item ? (item.call_id ?? 'tool_result_call_id') : '',
      name: item.type,
      result: JSON.stringify(item),
      providerSpecific: ApiModes.RESPONSE,
    };
  },

  fromAnnotation(
    item:
      | ResponseOutputText.FileCitation
      | ResponseOutputText.URLCitation
      | ResponseOutputText.ContainerFileCitation
      | ResponseOutputText.FilePath,
  ): Citation {
    switch (item.type) {
      case 'file_citation':
        return {
          type: 'file_location',
          fileId: item.file_id,
          fileName: item.filename,
          citedText: '',
          extra: {
            index: item.index, // The index of the file in the list of files. But what is the list of files?
          },
        } as Citation;
      case 'url_citation':
        return {
          type: 'web_location',
          url: item.url,
          title: item.title,
          citedText: '',
          startIndex: item.start_index,
          endIndex: item.end_index,
        } as Citation;
      case 'container_file_citation':
        return {
          type: 'file_location',
          fileId: item.file_id,
          fileName: item.filename,
          citedText: '',
          startIndex: item.start_index,
          endIndex: item.end_index,
          extra: {
            container_id: item.container_id,
          },
        } as Citation;
      case 'file_path':
        return {
          type: 'file_location',
          fileId: item.file_id,
          citedText: '',
          fileName: '',
          extra: {
            index: item.index, // The index of the file in the list of files. But what is the list of files?
          },
        } as Citation;
      default:
        assertNever(item);
    }
  },
};

const ContentPartResponseHelper = {
  toResponseItems(
    role: RoleType,
    content: string | ContentPart[],
  ): Array<ResponseInputItem | ResponseOutputItem> {
    if (typeof content === 'string') {
      return [
        ContentPartResponseHelper.toResponseMessage(role === 'tool' ? 'user' : role, [
          { type: 'text', text: content } as TextPart,
        ]),
      ];
    }

    const items: Array<ResponseInputItem | ResponseOutputItem> = [];
    const effectiveRole: Exclude<RoleType, 'tool'> = role === 'tool' ? 'user' : role;
    let currentMessageParts: Array<TextPart | ImagePart | DocumentPart | RefusalPart> = [];

    const flushMessage = () => {
      if (currentMessageParts.length === 0) {
        return;
      }
      items.push(ContentPartResponseHelper.toResponseMessage(effectiveRole, currentMessageParts));
      currentMessageParts = [];
    };

    for (const part of content) {
      if (
        part.type === 'tool_call' ||
        part.type === 'tool_result' ||
        (part.type === 'image' && role === 'assistant')
      ) {
        const standalone = ContentPartResponseHelper.toStandaloneResponseItem(part, role);
        if (standalone) {
          flushMessage();
          items.push(standalone);
        }
      } else if (
        part.type === 'text' ||
        part.type === 'refusal' ||
        part.type === 'image' ||
        part.type === 'document'
      ) {
        const messageCompatible = ContentPartResponseHelper.toResponseMessageContentPart(
          part,
          effectiveRole,
        );
        if (messageCompatible) {
          currentMessageParts.push(part);
        }
      }
    }

    flushMessage();
    return items;
  },

  toResponseMessage(
    role: Exclude<RoleType, 'tool'>,
    parts: Array<TextPart | ImagePart | DocumentPart | RefusalPart>,
  ): ResponseInputItem.Message | ResponseOutputMessage {
    const content = parts
      .map((part) => ContentPartResponseHelper.toResponseMessageContentPart(part, role))
      .filter((item) => item !== null);

    if (role === 'assistant') {
      return {
        type: 'message',
        role: 'assistant',
        status: 'completed',
        content,
      } as ResponseOutputMessage;
    }

    return {
      type: 'message',
      role: 'user',
      content,
    } as ResponseInputItem.Message;
  },

  toResponseMessageContentPart(
    part: TextPart | RefusalPart | ImagePart | DocumentPart,
    role: Exclude<RoleType, 'tool'>,
  ):
    | ResponseInputText
    | ResponseInputImage
    | ResponseInputFile
    | ResponseOutputText
    | ResponseOutputRefusal
    | null {
    switch (part.type) {
      case 'text': {
        if (role === 'assistant') {
          return {
            type: 'output_text',
            text: part.text,
            annotations: (part.citations ?? [])
              .map((c) => ContentPartResponseHelper.convertCitationToResponseAnnotation(c))
              .filter((a) => a !== null),
          };
        } else {
          return {
            type: 'input_text',
            text: part.text,
          };
        }
      }
      case 'refusal': {
        return role === 'assistant'
          ? {
              type: 'refusal',
              refusal: part.reason,
            }
          : null;
      }
      case 'image': {
        return role === 'assistant'
          ? null
          : ContentPartResponseHelper.convertToResponseInputImageContent(part);
      }
      case 'document': {
        return role === 'assistant'
          ? null
          : ContentPartResponseHelper.convertToResponseInputFileContent(part);
      }
      default:
        assertNever(part);
    }
  },

  toStandaloneResponseItem(
    part: ToolCallPart | ToolResultPart | ImagePart,
    role: RoleType,
  ): ResponseInputItem | ResponseOutputItem | null {
    switch (part.type) {
      case 'tool_call': {
        return {
          type: 'function_call',
          id: part.id,
          call_id: part.id,
          name: part.name,
          arguments: part.arguments,
          status: 'completed',
        } as ResponseFunctionToolCall;
      }
      case 'tool_result': {
        return {
          type: 'function_call_output',
          call_id: part.id,
          output: part.result ?? '',
          status: 'completed',
        } as ResponseInputItem.FunctionCallOutput;
      }
      case 'image': {
        if (
          role === 'assistant' &&
          (part.image.sourceType === 'base64' || part.image.sourceType === 'file_id')
        ) {
          return {
            type: 'image_generation_call',
            id: part.image.sourceType === 'file_id' ? part.image.fileId : randomUUID(),
            result: part.image.sourceType === 'base64' ? part.image.data : null,
            status: 'completed',
          } as ResponseOutputItem.ImageGenerationCall;
        } else {
          return null;
        }
      }
      default: {
        assertNever(part);
      }
    }
  },

  convertToResponseInputImageContent(part: ImagePart): ResponseInputImage {
    switch (part.image.sourceType) {
      case 'base64': {
        return {
          type: 'input_image',
          detail: 'auto',
          image_url: `data:${part.image.mimeType};base64,${part.image.data}`,
        };
      }
      case 'file_id': {
        return {
          type: 'input_image',
          detail: 'auto',
          file_id: part.image.fileId,
        };
      }
      case 'url': {
        return {
          type: 'input_image',
          detail: 'auto',
          image_url: part.image.url,
        };
      }
      default:
        assertNever(part.image);
    }
  },

  convertToResponseInputFileContent(part: DocumentPart): ResponseInputFile {
    const source = part.document;
    switch (source.sourceType) {
      case 'url': {
        return {
          type: 'input_file',
          file_url: source.url,
          filename: source.fileName,
        };
      }
      case 'base64': {
        return {
          type: 'input_file',
          file_data: source.data,
          filename: source.fileName,
        };
      }
      case 'file_id': {
        return {
          type: 'input_file',
          file_id: source.fileId,
          filename: source.fileName,
        };
      }
      default:
        assertNever(source);
    }
  },

  convertCitationToResponseAnnotation(
    citation: Citation,
  ):
    | ResponseOutputText.FileCitation
    | ResponseOutputText.URLCitation
    | ResponseOutputText.ContainerFileCitation
    | ResponseOutputText.FilePath
    | null {
    switch (citation.type) {
      case 'web_location': {
        return {
          type: 'url_citation',
          url: citation.url,
          title: citation.title ?? '',
          start_index: citation.startIndex ?? 0,
          end_index: citation.endIndex ?? 0,
        };
      }
      case 'file_location': {
        const fileId = citation.extra?.file_id;
        const filename = citation.fileName;
        const container_id = citation.extra?.container_id;
        if (!fileId || typeof fileId !== 'string') {
          return null;
        }
        if (container_id !== undefined && typeof container_id === 'string') {
          return {
            type: 'container_file_citation',
            file_id: fileId,
            filename: filename,
            start_index: citation.startIndex ?? 0,
            end_index: citation.endIndex ?? 0,
            container_id: container_id,
          };
        } else if (filename === '') {
          return {
            type: 'file_path',
            file_id: fileId,
            index: typeof citation.extra?.index === 'number' ? citation.extra?.index : 0,
          };
        } else {
          return {
            type: 'file_citation',
            file_id: fileId,
            filename: filename,
            index: typeof citation.extra?.index === 'number' ? citation.extra?.index : 0,
          };
        }
      }
      case 'others_location': {
        return null;
      }
      default:
        assertNever(citation);
    }
  },
};

export const ConvertHelper = {
  toContentParts: (item: ResponseInputItem | ResponseOutputItem): ContentPart[] =>
    ResponseContentPartHelper.toContentPartsFromResponseItem(item),
  toContentPart: (
    item:
      | Exclude<
          ResponseInputItem,
          EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage
        >
      | Exclude<ResponseOutputItem, ResponseOutputMessage>
      | ResponseOutputText
      | ResponseInputText
      | ResponseOutputRefusal
      | ResponseInputImage
      | ResponseInputImageContent
      | ResponseInputFile
      | ResponseInputFileContent,
  ): ContentPart | null => ResponseContentPartHelper.fromResponseObject(item),
  toResponseItems: (
    role: RoleType,
    content: string | ContentPart[],
  ): Array<ResponseInputItem | ResponseOutputItem> =>
    ContentPartResponseHelper.toResponseItems(role, content),
};

function assertNever(x: never): never {
  throw new Error(`Unexpected object: ${x}`);
}
