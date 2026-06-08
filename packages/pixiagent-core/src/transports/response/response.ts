import { OpenAI } from 'openai/client';
import type {
  Response,
  ResponseCreateParamsStreaming,
  ResponseInputItem,
  ResponseOutputItem,
  ToolChoiceOptions,
  ResponseErrorEvent,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import { ApiModes, SessionMessage, ResponseApiMessage, ModelStopReasons } from '../../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../../errors';
import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ProviderTransport,
  StreamCallbacks,
} from '../base';
import { ResponseConversionHelper } from './response-conversion';
import { ResponseStreamProcessor } from './response-stream';

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
    const parts = rawMsg.content.flatMap(ResponseConversionHelper.toContentParts);
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
    const items = ResponseConversionHelper.toResponseItems(msg);
    const responseApiMessage: ResponseApiMessage = {
      messageId: msg.messageId,
      type: 'response_api_message',
      role: msg.role,
      content: items,
      metadata: msg.metadata,
      ...(msg.modelResponseInfo ? { modelResponseInfo: msg.modelResponseInfo } : {}),
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

      const streamProcessor = new ResponseStreamProcessor(
        this.dialectResolver as DialectResolver<
          ResponseApiMessage,
          ResponseStreamEvent,
          ResponseCreateParamsStreaming,
          Response
        > | undefined,
        this.client.baseURL,
      );
      const response = await streamProcessor.process(stream as AsyncIterable<ResponseStreamEvent>, callbacks);

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

