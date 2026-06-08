import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ProviderTransport,
  StreamCallbacks,
} from '../base';
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletion,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionReasoningEffort,
} from 'openai/resources/chat/completions';
import {
  ApiModes,
  ChatCompletionApiMessage,
  SessionMessage,
  ModelStopReasons,
} from '../../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../../errors';
import { OpenAI } from 'openai/client';
import { ChatCompletionMessageConverter } from './chat_completion.converters';
import { ChatCompletionStreamProcessor } from './chat_completion.stream';

export class ChatCompletionTransport extends ProviderTransport<ChatCompletionApiMessage> {
  readonly client: OpenAI;
  private static readonly OFFICIAL_BASE_URL = 'https://api.openai.com/v1';
  private readonly configuredBaseUrl?: string;
  private readonly messageConverter = new ChatCompletionMessageConverter();
  private readonly streamProcessor = new ChatCompletionStreamProcessor(this.dialectResolver as never);

  private static normalizeBaseUrl(baseUrl?: string): string | undefined {
    if (!baseUrl) return baseUrl;
    const normalized = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    if (normalized.toLowerCase().endsWith('/chat/completions')) {
      return normalized.slice(0, -'/chat/completions'.length);
    }
    return normalized;
  }

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      ChatCompletionApiMessage,
      ChatCompletionChunk.Choice.Delta,
      ChatCompletionCreateParamsStreaming,
      ChatCompletion
    >,
  ) {
    super(ApiModes.COMPLETIONS, dialectResolver);
    this.configuredBaseUrl = ChatCompletionTransport.normalizeBaseUrl(baseUrl);
    this.client = new OpenAI({
      baseURL: this.configuredBaseUrl,
      apiKey: apiKey,
    });
  }

  getApiMode(): ApiModes {
    return ApiModes.COMPLETIONS;
  }

  convertFromRawMessage(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const message = this.messageConverter.convertFromRawMessage(rawMsg);
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(message, rawMsg) : message;
  }

  convertToRawMessage(msg: SessionMessage): ChatCompletionApiMessage | ChatCompletionApiMessage[] {
    return this.messageConverter.convertToRawMessage(msg);
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

  private buildModelResponse(
    response: ChatCompletion,
  ): Omit<ChatCompletionApiMessage, 'messageId'> {
    const responseMessage = response.choices[0]?.message;
    return {
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: responseMessage,
      modelResponseInfo: {
        responseId: response.id,
        responseModel: response.model,
        stopReason: this.getStopReason(
          this.dialectResolver?.extractFromResponse('stop_reason', response) ??
            response.choices[0]?.finish_reason,
        ),
        refusal:
          responseMessage.role === 'assistant' && 'refusal' in responseMessage
            ? ((responseMessage as ChatCompletionAssistantMessageParam).refusal ?? undefined)
            : undefined,
        usage: response.usage
          ? {
              inputTokens: response.usage.prompt_tokens,
              outputTokens: response.usage.completion_tokens,
              totalTokens: response.usage.total_tokens,
              reasoningTokens:
                this.dialectResolver?.extractFromResponse('reasoning_tokens', response) ??
                response.usage.completion_tokens_details?.reasoning_tokens ??
                undefined,
              cacheReadTokens:
                this.dialectResolver?.extractFromResponse('cache_read_tokens', response) ??
                response.usage.prompt_tokens_details?.cached_tokens ??
                undefined,
              cacheCreatedTokens:
                this.dialectResolver?.extractFromResponse('cache_created_tokens', response) ??
                undefined,
              inputTokenDetails: response.usage.prompt_tokens_details
                ? { ...response.usage.prompt_tokens_details }
                : undefined,
              outputTokenDetails: response.usage.completion_tokens_details
                ? { ...response.usage.completion_tokens_details }
                : undefined,
            }
          : undefined,
      },
      metadata: {
        pixiagent_response_id: response.id,
        pixiagent_response_finish_reason: response.choices[0]?.finish_reason,
      },
    };
  }

  async generate(
    options: ModelOptions,
    messages: Array<ChatCompletionApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<ChatCompletionApiMessage, 'messageId'>> {
    const params = this.getChatCompletionStreamParams(options, messages);
    try {
      const stream = await this.client.chat.completions.create(
        { ...params, stream: true },
        this.getStreamRequestOptions(requestOptions),
      );
      const response = await this.streamProcessor.process(
        stream as AsyncIterable<ChatCompletionChunk>,
        callbacks,
      );
      return this.buildModelResponse(response);
    } catch (error) {
      throw this.wrapRequestError(error, requestOptions);
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
      this.client.baseURL ?? ChatCompletionTransport.OFFICIAL_BASE_URL,
      requestOptions?.timeout,
      undefined,
      error,
    );
  }

  private getStopReason(finishReason: string): ModelStopReasons {
    switch (finishReason) {
      case 'stop':
        return ModelStopReasons.STOP;
      case 'tool_calls':
      case 'function_call':
        return ModelStopReasons.TOOL_CALL;
      case 'length':
        return ModelStopReasons.MAX_TOKENS;
      case 'content_filter':
        return ModelStopReasons.REFUSAL;
      default:
        return ModelStopReasons.OTHERS;
    }
  }

  /**
   * Determines the reasoning_effort value based on model capabilities and thinkEffort option.
   *
   * - O-series models (o1, o3, o4-mini, …) and gpt-5.x models support reasoning_effort.
   * - gpt-5.<digit> models (e.g. gpt-5.1, gpt-5.4-nano) support the 'none' value.
   * - O-series and gpt-5-<name> models (e.g. gpt-5-pro) do NOT support 'none'.
   * - Non-reasoning models → undefined (field omitted from params).
   */
  private getReasoningEffort(
    model: string,
    thinkEffort?: ModelOptions['thinkEffort'],
  ): ChatCompletionReasoningEffort | undefined {
    const baseName = model.includes('/') ? model.split('/').pop()! : model;
    const isOSeries = /^o\d/i.test(baseName);
    const isGpt5x = /^gpt-5/i.test(baseName);
    if (!isOSeries && !isGpt5x) return undefined;

    // gpt-5.<digit> models support 'none'; o-series and gpt-5-<name> do not
    const supportsNone = /^gpt-5\.\d/i.test(baseName);
    switch (thinkEffort) {
      case 'disable':
        return supportsNone ? 'none' : null;
      case 'low':
        return 'low';
      case 'medium':
        return 'medium';
      case 'high':
        return 'high';
      case 'extreme':
        return 'xhigh';
      default:
        return null; // use model default
    }
  }

  private getChatCompletionStreamParams(
    options: ModelOptions,
    messages: Array<ChatCompletionApiMessage>,
  ): ChatCompletionCreateParamsStreaming {
    const inputs: ChatCompletionMessageParam[] = [];
    if (options.systemPrompt) {
      inputs.push({
        role: 'developer',
        content: options.systemPrompt,
      });
    }
    inputs.push(...messages.map((m) => m.content));

    const params: ChatCompletionCreateParamsStreaming = {
      model: options.model,
      messages: inputs,
      max_completion_tokens: options.maxTokens,
      temperature: options.temperature,
      metadata: options.metadata,
      reasoning_effort: this.getReasoningEffort(options.model, options.thinkEffort),
      store: false,
      stream: true,
      tools: options.tools?.map((t) => ({
        type: 'function',
        function: t,
      })),
    };

    return this.dialectResolver
      ? (this.dialectResolver.manipulateOptions(
          options,
          params,
        ) as ChatCompletionCreateParamsStreaming)
      : params;
  }
}
