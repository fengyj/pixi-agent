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
  ResponseStreamEvent,
  ResponseFunctionWebSearch,
  ResponseFileSearchToolCall,
  ResponseCodeInterpreterToolCall,
  ResponseFunctionToolCallOutputItem,
} from 'openai/resources/responses/responses';
import {
  ApiModes,
  Citation,
  ContentPart,
  DocumentPart,
  ImagePart,
  RefusalPart,
  SessionMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ResponseApiMessage,
  ThinkingPart,
  ModelStopReasons,
  ServerToolUsePart,
} from '../../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../../errors';
import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ProviderTransport,
  StreamCallbacks,
  StreamDataExtractor,
} from '../base';
import { randomUUID } from 'crypto';
import { assertNever } from '../../utils';

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
      ResponseStreamEvent,
      ResponseCreateParamsStreaming,
      Response
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
    const items = ConvertHelper.toResponseItems(msg);
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
        params,
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
              (delta) => (delta.text === '' ? null : { type: 'text', text: delta.text }),
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
              (delta) => (delta.refusal === '' ? null : ConvertHelper.toContentPart(delta)),
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
              (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
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
              (delta) => (delta.text === '' ? null : { type: 'thinking', content: delta.text }),
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
              (delta) => (delta.arguments === '' ? null : ConvertHelper.toContentPart(delta)),
            );
            break;
          }
          case 'response.function_call_arguments.done':
            break;
          case 'response.audio.delta':
          case 'response.audio.done':
          case 'response.audio.transcript.delta':
          case 'response.audio.transcript.done':
          case 'response.code_interpreter_call_code.delta':
          case 'response.code_interpreter_call_code.done':
          case 'response.code_interpreter_call.completed':
          case 'response.code_interpreter_call.in_progress':
          case 'response.code_interpreter_call.interpreting':
          case 'response.custom_tool_call_input.delta':
          case 'response.custom_tool_call_input.done':
          case 'response.file_search_call.completed':
          case 'response.file_search_call.in_progress':
          case 'response.file_search_call.searching':
          case 'response.image_generation_call.completed':
          case 'response.image_generation_call.generating':
          case 'response.image_generation_call.in_progress':
          case 'response.image_generation_call.partial_image':
          case 'response.mcp_call_arguments.delta':
          case 'response.mcp_call_arguments.done':
          case 'response.mcp_call.completed':
          case 'response.mcp_call.failed':
          case 'response.mcp_call.in_progress':
          case 'response.mcp_list_tools.completed':
          case 'response.mcp_list_tools.failed':
          case 'response.mcp_list_tools.in_progress':
          case 'response.web_search_call.completed':
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
    return ResponseContentPartHelper.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromOutputItem(item: ResponseOutputItem): ContentPart[] {
    return ResponseContentPartHelper.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromResponseItem(item: ResponseInputItem | ResponseOutputItem): ContentPart[] {
    const isResponseMessageItem = (
      data: ResponseInputItem | ResponseOutputItem,
    ): data is EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage => {
      return (
        data.type === 'message' ||
        (!data.type &&  // EasyInputMessage may not have 'type' field
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
      return ResponseContentPartHelper.fromResponseMessageItem(item);
    } else if (
      item.type === 'compaction' ||
      item.type === 'compaction_trigger' ||
      item.type === 'item_reference'
    ) {
      // These are control or metadata-only items and do not represent renderable
      // content that can be mapped to a ContentPart.
      return [];
    } else {
      const contentPart = ResponseContentPartHelper.fromResponseObject(item);
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

    return contents
      .map((c) => ResponseContentPartHelper.fromResponseObject(c))
      .filter((part) => part !== null);
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
        return ResponseContentPartHelper.fromResponseInputText(item);
      case 'output_text':
        return ResponseContentPartHelper.fromResponseOutputText(item);
      case 'refusal':
        return ResponseContentPartHelper.fromResponseOutputRefusal(item);
      case 'input_image':
        return ResponseContentPartHelper.fromResponseInputImage(item);
      case 'input_file':
        return ResponseContentPartHelper.fromResponseInputFile(item);
      case 'image_generation_call':
        return ResponseContentPartHelper.fromResponseImageGenerationCall(item);
      case 'function_call':
        return ResponseContentPartHelper.fromResponseFunctionCall(item);
      case 'function_call_output':
        return ResponseContentPartHelper.fromResponseFunctionCallOutput(item);
      case 'reasoning':
        return ResponseContentPartHelper.fromResponseReasoningItem(item);
      case 'computer_call':
      case 'custom_tool_call':
      case 'tool_search_call':
      case 'local_shell_call':
      case 'shell_call':
      case 'apply_patch_call':
        return ResponseContentPartHelper.fromResponseSpecificToolCall(item);
      case 'computer_call_output':
      case 'custom_tool_call_output':
      case 'tool_search_output':
      case 'local_shell_call_output':
      case 'shell_call_output':
      case 'apply_patch_call_output':
        return ResponseContentPartHelper.fromResponseSpecificToolOutput(item);
      // server tool calls (which includes the request argument and the response output)
      case 'file_search_call': // ResponseFileSearchToolCall
      case 'web_search_call': // ResponseFunctionWebSearch
      case 'code_interpreter_call': // ResponseCodeInterpreterToolCall
      case 'mcp_call': // ResponseInputItem.McpCall | ResponseOutputItem.McpCall
      case 'mcp_list_tools': // ResponseInputItem.McpListTools | ResponseOutputItem.McpListTools
        return ResponseContentPartHelper.fromResponseServerToolCall(item);
      case 'mcp_approval_request': // ResponseInputItem.McpApprovalRequest | ResponseOutputItem.McpApprovalRequest
      case 'mcp_approval_response': // ResponseInputItem.McpApprovalResponse | ResponseOutputItem.McpApprovalResponse
      case 'compaction': // ResponseCompactionItemParam | ResponseCompactionItem
      case 'compaction_trigger': // ResponseInputItem.CompactionTrigger
        return null; // todo: how to handle these types?
      case 'item_reference': // ResponseInputItem.ItemReference | ResponseOutputItem.ItemReference
      case null: // ResponseInputItem.ItemReference
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
      citations: item.annotations?.flatMap(ResponseContentPartHelper.fromAnnotation) ?? [],
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
        item.output
          .map((c) => ResponseContentPartHelper.fromResponseObject(c))
          .filter((part) => part !== undefined),
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
  ): ToolCallPart | ServerToolUsePart {
    if ('call_id' in item && item.call_id) {
      const { call_id, type, ...rest } = item;
      return {
        type: 'tool_call',
        id: call_id,
        name: type,
        arguments: JSON.stringify(rest),
        providerSpecific: ApiModes.RESPONSE,
      };
    } else {
      const { type, ...rest } = item;
      return {
        type: 'server_tool_use',
        name: type,
        data: JSON.stringify(rest),
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
  ): ToolResultPart | ServerToolUsePart {
    if ('call_id' in item && item.call_id) {
      const { call_id, type, ...rest } = item;
      return {
        type: 'tool_result',
        id: call_id,
        name: type,
        result: JSON.stringify(rest),
        providerSpecific: ApiModes.RESPONSE,
      };
    } else {
      const { type, ...rest } = item;
      return {
        type: 'server_tool_use',
        name: type,
        data: JSON.stringify(rest),
        providerSpecific: ApiModes.RESPONSE,
      };
    }
  },

  fromResponseServerToolCall(
    item:
      | ResponseFileSearchToolCall
      | ResponseFunctionWebSearch
      | ResponseCodeInterpreterToolCall
      | ResponseInputItem.McpCall
      | ResponseOutputItem.McpCall
      | ResponseInputItem.McpListTools
      | ResponseOutputItem.McpListTools,
  ): ServerToolUsePart {
    const { type, ...rest } = item;
    return {
      type: 'server_tool_use',
      name: type,
      data: JSON.stringify(rest),
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
  toResponseItems(message: SessionMessage): Array<ResponseInputItem | ResponseOutputItem> {
    if (message.role === 'assistant') {
      return ContentPartResponseHelper.toResponseOutputItems(
        message as Extract<SessionMessage, { role: 'assistant' }>,
      );
    } else {
      return ContentPartResponseHelper.toResponseInputItems(
        message as Extract<SessionMessage, { role: 'user' | 'tool' }>,
      );
    }
  },
  toResponseInputItems(
    message: Omit<SessionMessage, 'role'> & { role: 'user' | 'tool' },
  ): ResponseInputItem[] {
    if (typeof message.content === 'string') {
      return [ContentPartResponseHelper.toInputMessage(message.content)];
    }
    const items = Array<
      | Exclude<
          ResponseInputItem,
          ResponseOutputMessage | EasyInputMessage | ResponseInputItem.Message
        >
      | ResponseInputText
      | ResponseInputImage
      | ResponseInputFile
    >();
    for (const part of message.content) {
      switch (part.type) {
        case 'text':
          items.push(ContentPartResponseHelper.toInputText(part));
          break;
        case 'image': {
          items.push(ContentPartResponseHelper.toInputImage(part));
          break;
        }
        case 'document':
          items.push(ContentPartResponseHelper.toInputFile(part));
          break;
        case 'tool_call':
          items.push(ContentPartResponseHelper.toToolCall(part as ToolCallPart));
          break;
        case 'tool_result':
          items.push(ContentPartResponseHelper.toToolResult(part as ToolResultPart));
          break;
        case 'server_tool_use':
          items.push(
            ContentPartResponseHelper.toItemFromServerToolUseForInput(part as ServerToolUsePart),
          );
          break;
        case 'audio':
        case 'video':
        case 'refusal':
        case 'thinking':
          break;
        default:
          assertNever(part);
      }
    }

    const itemsForMessage: Array<ResponseInputText | ResponseInputImage | ResponseInputFile> = [];
    const inputItems: Array<ResponseInputItem> = [];

    const generateMessage = (): void => {
      if (itemsForMessage.length > 0) {
        inputItems.push(ContentPartResponseHelper.toInputMessage(itemsForMessage));
        itemsForMessage.length = 0;
      }
    };

    for (const item of items) {
      switch (item.type) {
        case 'input_text':
        case 'input_image':
        case 'input_file':
          itemsForMessage.push(item);
          break;
        default:
          generateMessage();
          inputItems.push(item);
      }
    }
    return inputItems;
  },
  toResponseOutputItems(
    message: Omit<SessionMessage, 'role'> & { role: 'assistant' },
  ): ResponseOutputItem[] {
    if (typeof message.content === 'string') {
      return [ContentPartResponseHelper.toOutputMessage(message.messageId, message.content)];
    }
    const items = Array<
      | Exclude<ResponseOutputItem, ResponseOutputMessage>
      | ResponseOutputText
      | ResponseOutputRefusal
    >();
    for (const part of message.content) {
      switch (part.type) {
        case 'text':
          items.push(ContentPartResponseHelper.toOutputText(part));
          break;
        case 'refusal':
          items.push(ContentPartResponseHelper.toRefusal(part));
          break;
        case 'thinking':
          items.push(ContentPartResponseHelper.toThinking(part));
          break;
        case 'image': {
          switch (part.image.sourceType) {
            case 'base64':
              items.push(
                ContentPartResponseHelper.toImageGenerationCall(
                  part as ImagePart & { image: { sourceType: 'base64' } },
                ),
              );
              break;
            default:
              break;
          }
          break;
        }
        case 'tool_call':
          items.push(ContentPartResponseHelper.toToolCall(part as ToolCallPart));
          break;
        case 'tool_result':
          items.push(ContentPartResponseHelper.toToolResultForOutputItem(part as ToolResultPart));
          break;
        case 'server_tool_use':
          items.push(ContentPartResponseHelper.toItemFromServerToolUse(part as ServerToolUsePart));
          break;
        case 'audio':
        case 'document':
        case 'video':
          break;
        default:
          assertNever(part);
      }
    }

    const itemsForMessage: Array<ResponseOutputText | ResponseOutputRefusal> = [];
    const itemsForThinking: Array<ResponseReasoningItem> = [];
    const outputItems: Array<ResponseOutputItem> = [];

    const generateMessage = (): void => {
      if (itemsForMessage.length > 0) {
        outputItems.push(
          ContentPartResponseHelper.toOutputMessage(
            `${message.messageId}_${outputItems.length}`,
            itemsForMessage,
          ),
        );
        itemsForMessage.length = 0;
      }
    };
    const generateThinkingItem = (): void => {
      if (itemsForThinking.length > 0) {
        outputItems.push({
          id: `${message.messageId}_${outputItems.length}`,
          type: 'reasoning',
          summary: itemsForThinking.flatMap((i) => i.summary),
        } as ResponseReasoningItem);
        itemsForThinking.length = 0;
      }
    };

    for (const item of items) {
      switch (item.type) {
        case 'output_text':
        case 'refusal':
          generateThinkingItem();
          itemsForMessage.push(item);
          break;
        case 'reasoning':
          generateMessage();
          itemsForThinking.push(item);
          break;
        default:
          generateThinkingItem();
          generateMessage();
          outputItems.push(item);
      }
    }
    return outputItems;
  },
  toOutputMessage(
    id: string,
    items: string | Array<ResponseOutputText | ResponseOutputRefusal>,
  ): ResponseOutputMessage {
    if (typeof items === 'string') {
      return {
        id: id,
        type: 'message',
        role: 'assistant',
        content: [
          {
            type: 'output_text',
            text: items,
            annotations: [],
          } as ResponseOutputText,
        ],
        status: 'completed',
      };
    }
    return {
      id: id,
      type: 'message',
      role: 'assistant',
      content: items,
      status: 'completed',
    };
  },
  toInputMessage(
    content: string | Array<ResponseInputText | ResponseInputImage | ResponseInputFile>,
  ): EasyInputMessage | ResponseInputItem.Message {
    if (typeof content === 'string') {
      return {
        type: 'message',
        role: 'user',
        content: content,
      };
    }
    return {
      type: 'message',
      role: 'user',
      content,
    };
  },

  toOutputText(part: TextPart): ResponseOutputText {
    return {
      type: 'output_text',
      text: part.text,
      annotations: (part.citations ?? [])
        .map((c) => ContentPartResponseHelper.toAnnotation(c))
        .filter((a) => a !== null),
    };
  },
  toRefusal(part: RefusalPart): ResponseOutputRefusal {
    return {
      type: 'refusal',
      refusal: part.reason,
    };
  },
  toItemFromServerToolUse(
    part: ServerToolUsePart,
  ):
    | ResponseOutputText
    | ResponseFileSearchToolCall
    | ResponseFunctionWebSearch
    | ResponseCodeInterpreterToolCall
    | ResponseComputerToolCall
    | ResponseCustomToolCall
    | ResponseOutputItem.McpCall
    | ResponseOutputItem.McpListTools
    | ResponseComputerToolCallOutputItem
    | ResponseCustomToolCallOutputItem
    | ResponseToolSearchOutputItem
    | ResponseOutputItem.LocalShellCallOutput
    | ResponseFunctionShellToolCallOutput
    | ResponseApplyPatchToolCallOutput
    | ResponseToolSearchCall
    | ResponseOutputItem.LocalShellCall
    | ResponseFunctionShellToolCall
    | ResponseApplyPatchToolCall {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      try {
        return {
          ...JSON.parse(part.data ?? '{}'),
          type: part.name,
          call_id: part.id,
        };
      } catch {
        // empty
      }
    }
    return {
      type: 'output_text',
      text: `Tool use: ${part.name} with data ${part.data}`,
      annotations: [],
    };
  },

  toItemFromServerToolUseForInput(
    part: ServerToolUsePart,
  ):
    | ResponseInputText
    | ResponseFileSearchToolCall
    | ResponseFunctionWebSearch
    | ResponseCodeInterpreterToolCall
    | ResponseComputerToolCall
    | ResponseCustomToolCall
    | ResponseInputItem.McpCall
    | ResponseInputItem.McpListTools
    | ResponseInputItem.ComputerCallOutput
    | ResponseCustomToolCallOutput
    | ResponseToolSearchOutputItemParam
    | ResponseInputItem.LocalShellCallOutput
    | ResponseInputItem.ShellCallOutput
    | ResponseInputItem.ApplyPatchCallOutput
    | ResponseInputItem.ToolSearchCall
    | ResponseInputItem.LocalShellCall
    | ResponseInputItem.ShellCall
    | ResponseInputItem.ApplyPatchCall {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      try {
        return {
          ...JSON.parse(part.data ?? '{}'),
          type: part.name,
          call_id: part.id,
        };
      } catch {
        // empty
      }
    }
    return {
      type: 'input_text',
      text: `Tool use: ${part.name} with data ${part.data}`,
    };
  },
  toImageGenerationCall(
    part: ImagePart & { image: { sourceType: 'base64' } },
  ): ResponseOutputItem.ImageGenerationCall {
    return {
      type: 'image_generation_call',
      id: randomUUID(),
      result: part.image.data,
      status: part.image.data === '' ? 'failed' : 'completed',
    } as ResponseOutputItem.ImageGenerationCall;
  },
  toToolCall(part: ToolCallPart): ResponseFunctionToolCall {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      return {
        ...JSON.parse(part.arguments),
        type: part.name,
        call_id: part.id,
      };
    }
    return {
      type: 'function_call',
      id: part.id,
      call_id: part.id,
      name: part.name,
      arguments: part.arguments,
    } as ResponseFunctionToolCall;
  },
  toToolResult(part: ToolResultPart):
    | ResponseInputItem.FunctionCallOutput
    // they are provider specific function calls below.
    | ResponseInputItem.ComputerCallOutput
    | ResponseCustomToolCallOutput
    | ResponseToolSearchOutputItemParam
    | ResponseInputItem.LocalShellCallOutput
    | ResponseInputItem.ShellCallOutput
    | ResponseInputItem.ApplyPatchCallOutput {
    try {
      const parsedResult = !part.result || part.result === '' ? null : JSON.parse(part.result);

      if (part.providerSpecific === ApiModes.RESPONSE) {
        return {
          ...parsedResult,
          type: part.name,
          call_id: part.id,
        };
      }

      if (parsedResult && Array.isArray(parsedResult)) {
        const convertable = parsedResult.every((c) => {
          if (!('type' in c)) return false;
          switch (c.type) {
            case 'text':
              return 'text' in c && typeof c.text === 'string';
            case 'image':
              return (
                ('image_url' in c && typeof c.image_url === 'string') ||
                ('file_id' in c && typeof c.file_id === 'string')
              );
            case 'document':
              return (
                ('file_url' in c && typeof c.file_url === 'string') ||
                ('file_data' in c && typeof c.file_data === 'string') ||
                ('file_id' in c && typeof c.file_id === 'string')
              );
            default:
              return false;
          }
        });
        if (convertable) {
          const outputParts = parsedResult.map((c) => c as TextPart | ImagePart | DocumentPart);
          return {
            type: 'function_call_output',
            call_id: part.id,
            output: outputParts
              .map((c) => {
                switch (c.type) {
                  case 'text':
                    return ContentPartResponseHelper.toInputText(c);
                  case 'image':
                    return ContentPartResponseHelper.toInputImage(c);
                  case 'document':
                    return ContentPartResponseHelper.toInputFile(c);
                  default:
                    assertNever(c);
                }
              })
              .filter((item) => item !== null) as Array<ResponseInputText | ResponseInputImage>,
            status: 'completed',
          };
        }
      }
    } catch {
      // empty
    }
    return {
      type: 'function_call_output',
      call_id: part.id,
      output: part.result ?? '',
      status: 'completed',
    };
  },
  toToolResultForOutputItem(part: ToolResultPart):
    | ResponseFunctionToolCallOutputItem
    // they are provider specific function calls below.
    | ResponseComputerToolCallOutputItem
    | ResponseCustomToolCallOutputItem
    | ResponseToolSearchOutputItem
    | ResponseOutputItem.LocalShellCallOutput
    | ResponseFunctionShellToolCallOutput
    | ResponseApplyPatchToolCallOutput {
    const { id, ...rest } = ContentPartResponseHelper.toToolResult(part);
    switch (rest.type) {
      case 'computer_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status ?? 'completed',
          acknowledged_safety_checks: rest.acknowledged_safety_checks ?? undefined,
        };
      case 'apply_patch_call_output':
        return {
          ...rest,
          id: id ?? '',
        };
      case 'local_shell_call_output':
        return {
          ...rest,
          id: id ?? '',
        };
      case 'shell_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          max_output_length: rest.max_output_length ?? null,
        };
      case 'tool_search_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          call_id: rest.call_id ?? '',
          execution: rest.execution ?? 'server',
        };
      case 'function_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          output:
            typeof rest.output === 'string'
              ? rest.output
              : rest.output.map((c) => {
                  switch (c.type) {
                    case 'input_text':
                      return c as ResponseInputText;
                    case 'input_image':
                      return c as ResponseInputImage;
                    case 'input_file':
                      return c as ResponseInputFile;
                    default:
                      assertNever(c);
                  }
                }),
        };
      case 'custom_tool_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: 'completed',
        };
    }
  },
  toThinking(part: ThinkingPart): ResponseReasoningItem {
    return {
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: part.content }],
      id: '',
    };
  },
  toInputText(part: TextPart): ResponseInputText {
    return {
      type: 'input_text',
      text: part.text,
    };
  },
  toInputImage(part: ImagePart): ResponseInputImage {
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

  toInputFile(part: DocumentPart): ResponseInputFile {
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

  toAnnotation(
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
  toContentParts: ResponseContentPartHelper.toContentPartsFromResponseItem,
  toContentPart: ResponseContentPartHelper.fromResponseObject,
  toResponseInputItems: ContentPartResponseHelper.toResponseInputItems,
  toResponseOutputItems: ContentPartResponseHelper.toResponseOutputItems,
  toResponseItems: ContentPartResponseHelper.toResponseItems,
};
