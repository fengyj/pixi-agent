import Anthropic from '@anthropic-ai/sdk';
import type { MessageParam, Tool as AnthropicTool } from '@anthropic-ai/sdk/resources/messages';
import type {
  Message,
  MessageCreateParamsStreaming,
  RawContentBlockDelta,
} from '@anthropic-ai/sdk/resources/messages/messages';
import { ProviderTransport, ModelOptions, StreamCallbacks, DialectResolver, ModelRequestOptions } from '../base';
import { AnthropicApiMessage, ApiModes, SessionMessage } from '../../message';
import { PixiAgentErrorBuilder, ErrorGuards } from '../../errors';
import { AnthropicMessageConverter } from './anthropic.converters';
import { AnthropicStreamProcessor } from './anthropic.stream';

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
          : inner.content.map((block) => AnthropicMessageConverter.toParts(block as never)).flat(),
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
          : msg.content.map((part) => AnthropicMessageConverter.toBlockParam(part)),
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

  private async generateStream(
    params: MessageCreateParamsStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<AnthropicApiMessage, 'messageId'>> {
    const stream = await this.client.messages.create(
      params,
      this.getStreamRequestOptions(requestOptions),
    );

    const streamProcessor = new AnthropicStreamProcessor(
      this.dialectResolver as
        | import('../base').DialectResolver<
            AnthropicApiMessage,
            RawContentBlockDelta,
            MessageCreateParamsStreaming,
            Message
          >
        | undefined,
      this.client.baseURL ?? AnthropicTransport.OFFICIAL_BASE_URL,
    );

    return await streamProcessor.process(stream, callbacks);
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

}
