import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionStreamParams,
} from 'openai/resources/chat/completions';
import {
  ApiModes,
  AnthropicApiMessage,
  ChatCompletionApiMessage,
  SessionMessage,
  ThinkingPart,
  ContentPart,
} from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions, StreamDataExtractor } from '../base';
import type {
  Message,
  MessageParam,
  MessageStreamParams,
  RawContentBlockDelta,
  ThinkingBlockParam,
} from '@anthropic-ai/sdk/resources/messages/messages';

export class DeepSeekApiModeResolver extends ApiModeResolver {
  getApiMode(_model: string, baseUrl?: string): ApiModes | undefined {
    if (!baseUrl) return ApiModes.COMPLETIONS;
    else if (baseUrl.toLowerCase() === 'https://api.deepseek.com') return ApiModes.COMPLETIONS;
    else if (baseUrl.toLowerCase() === 'https://api.deepseek.com/anthropic')
      return ApiModes.ANTHROPIC;
    return undefined;
  }
  getBaseUrl(_model: string, apiMode?: ApiModes): string | undefined {
    if (!apiMode || apiMode === ApiModes.COMPLETIONS) return 'https://api.deepseek.com';
    else if (apiMode === ApiModes.ANTHROPIC) return 'https://api.deepseek.com/anthropic';
    else return undefined;
  }
}

/**
 * Dialect resolver for DeepSeek via the OpenAI Chat Completions API (https://api.deepseek.com).
 * Handles the provider-specific `reasoning_content` field for thinking/reasoning.
 */
export class DeepSeekChatDialectResolver implements DialectResolver<
  ChatCompletionApiMessage,
  ChatCompletionChunk.Choice.Delta,
  ChatCompletionStreamParams,
  ChatCompletion
> {
  // Models that doesn't support the reasoning_effort parameter
  private static readonly NON_REASONING_MODELS = ['deepseek-chat'];

  match(_model: string, baseUrl: string): boolean {
    return baseUrl.toLowerCase() === 'https://api.deepseek.com';
  }

  manipulateOptions(
    options: ModelOptions,
    parameters: ChatCompletionStreamParams,
  ): ChatCompletionStreamParams {
    // DeepSeek doesn't support the 'developer' role; convert to 'system'
    parameters.messages = parameters.messages.map((msg) =>
      msg.role === 'developer' ? { ...msg, role: 'system' } : msg,
    ) as ChatCompletionMessageParam[];

    if (!options.thinkEffort) return parameters;
    if (DeepSeekChatDialectResolver.NON_REASONING_MODELS.includes(options.model)) {
      parameters.reasoning_effort = undefined;
      return parameters;
    }
    // DeepSeek reasoning_effort: 'none' | 'high' | 'max'
    const effortMap: Record<string, string | null> = {
      disable: 'none',
      low: 'high',
      medium: 'high',
      high: 'high',
      extreme: 'max',
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (parameters as any).extra_body = {
      thinking: {
        type: options.thinkEffort && options.thinkEffort === 'disable' ? 'disabled' : 'enabled',
      },
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (parameters as any).reasoning_effort = effortMap[options.thinkEffort] ?? null;
    return parameters;
  }

  manipulateRawMessage(
    rawMsg: ChatCompletionApiMessage,
    msg?: SessionMessage,
  ): ChatCompletionApiMessage {
    if (!msg || rawMsg.role !== 'assistant' || !(msg.content instanceof Array)) return rawMsg;
    const thinkingParts = msg.content.filter((p): p is ThinkingPart => p.type === 'thinking');
    if (thinkingParts.length === 0) return rawMsg;
    // Write the accumulated thinking content back as reasoning_content
    const updatedInner = {
      ...rawMsg.content,
      reasoning_content: thinkingParts.map((p) => p.content).join(''),
    } as ChatCompletionAssistantMessageParam & { reasoning_content?: string };
    return { ...rawMsg, content: updatedInner };
  }

  manipulateMessage(msg: SessionMessage, rawMsg: ChatCompletionApiMessage): SessionMessage {
    const inner = rawMsg.content;
    if (inner.role !== 'assistant') return msg;
    const reasoningContent: string | undefined = (
      inner as ChatCompletionAssistantMessageParam & { reasoning_content?: string }
    ).reasoning_content;
    if (!reasoningContent) return msg;
    const thinkingPart: ThinkingPart = { type: 'thinking', content: reasoningContent };
    return { ...msg, content: ContentPart.concat([thinkingPart], msg.content) };
  }

  async extractFromDelta<T extends Record<string, unknown>>(
    data: string,
    delta: ChatCompletionChunk.Choice.Delta,
    streamDataExtractor: StreamDataExtractor<T>,
  ): Promise<void> {
    const getMessageObj = (acc: T): Record<string, unknown> | undefined => {
      if (!('choices' in acc && Array.isArray(acc.choices) && acc.choices.length > 0))
        return undefined;
      if (!('message' in acc.choices[0] && typeof acc.choices[0].message === 'object'))
        return undefined;
      return acc.choices[0].message;
    };

    if (data === 'reasoning') {
      if (
        'reasoning_content' in delta &&
        typeof delta.reasoning_content === 'string' &&
        delta.reasoning_content.length > 0
      ) {
        const reasoningUpdater: (acc: T, newData: string) => void = (acc, newData) => {
          const message = getMessageObj(acc);
          if (!message) return;
          if (!('reasoning_content' in message && typeof message.reasoning_content === 'string')) {
            (message as Record<string, unknown> & { reasoning_content: string }).reasoning_content =
              '';
          }
          message.reasoning_content += newData;
        };
        await streamDataExtractor.accumulate(
          { key: 'reasoning_content', value: delta.reasoning_content },
          reasoningUpdater,
          (_existing, newData, accumulated) => {
            if (!accumulated) return;
            reasoningUpdater(accumulated, newData);
          },
          (data) => {
            if (!data || data.length === 0) {
              return null;
            }
            return { type: 'thinking' as const, content: data }
          },
        );
      }
    }
  }

  extractFromResponse(
    data: 'cache_read_tokens' | 'cache_created_tokens' | string,
    response: ChatCompletion,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): any {
    if (data === 'cache_read_tokens') {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (response.usage as any)?.prompt_cache_hit_tokens ?? undefined;
    }
    return undefined;
  }
}

/**
 * Dialect resolver for DeepSeek via the Anthropic Messages API (https://api.deepseek.com/anthropic).
 * Handles thinking blocks in content using the Anthropic thinking block format.
 */
export class DeepSeekAnthropicDialectResolver implements DialectResolver<
  AnthropicApiMessage,
  RawContentBlockDelta,
  MessageStreamParams,
  Message
> {
  match(model: string, baseUrl: string): boolean {
    return baseUrl.toLowerCase() === 'https://api.deepseek.com/anthropic';
  }

  // Models that support extended thinking
  private static readonly REASONING_MODELS = ['deepseek-reasoner'];

  manipulateOptions(options: ModelOptions, parameters: MessageStreamParams): MessageStreamParams {
    if (!options.thinkEffort || options.thinkEffort === 'disable') return parameters;
    if (!DeepSeekAnthropicDialectResolver.REASONING_MODELS.includes(options.model))
      return parameters;
    // DeepSeek Anthropic uses output_config.effort for thinking strength.
    const effortMap: Record<string, 'high' | 'max'> = {
      low: 'high',
      medium: 'high',
      high: 'high',
      extreme: 'max',
    };
    const effort = effortMap[options.thinkEffort];
    if (effort) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (parameters as any).thinking = { type: 'enabled' };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (parameters as any).output_config = { effort };
    }
    return parameters;
  }

  // Block types not supported by DeepSeek's Anthropic-compatible API.
  private static readonly UNSUPPORTED_BLOCK_TYPES = new Set([
    'image',
    'document',
    'search_result',
    'server_tool_use',
    'web_search_tool_result',
    'web_fetch_tool_result',
    'code_execution_tool_result',
    'bash_code_execution_tool_result',
    'text_editor_code_execution_tool_result',
    'tool_search_tool_result',
    'container_upload',
  ]);

  manipulateRawMessage(rawMsg: AnthropicApiMessage, msg?: SessionMessage): AnthropicApiMessage {
    const innerContent = rawMsg.content;
    // Step 1: filter out content block types that DeepSeek doesn't support (image, document, etc.).
    // Step 2: for assistant messages, inject unsigned thinking blocks (DeepSeek doesn't use signatures).
    //
    // We only clone innerContent when we actually need to mutate it.
    // When msg is undefined, innerContent wasn't produced from a SessionMessage conversion, so we must
    // clone before mutating to avoid side-effects on the original object.

    const rawContent =
      typeof innerContent.content === 'string'
        ? [{ type: 'text' as const, text: innerContent.content }]
        : // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (innerContent.content as any[]);

    // Collect unsupported block indices to know whether filtering is needed.
    const filteredContent = rawContent.filter(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (b: any) => !DeepSeekAnthropicDialectResolver.UNSUPPORTED_BLOCK_TYPES.has(b.type),
    );
    const needsFilter = filteredContent.length !== rawContent.length;

    // Collect unsigned thinking parts to inject (assistant only).
    const unsignedThinkingParts =
      innerContent.role === 'assistant' && msg?.content instanceof Array
        ? msg.content.filter((p): p is ThinkingPart => p.type === 'thinking' && !p.signature)
        : [];
    const needsThinking = unsignedThinkingParts.length > 0;

    if (!needsFilter && !needsThinking) return rawMsg;

    // Clone only when we know a mutation is needed and msg is absent (shared reference risk).
    const base: MessageParam = msg ? innerContent : structuredClone(innerContent);

    const thinkingBlocks: ThinkingBlockParam[] = unsignedThinkingParts.map((p) => ({
      type: 'thinking' as const,
      thinking: p.content,
      signature: '',
    }));

    const updatedInner: MessageParam = {
      ...base,
      content: [...thinkingBlocks, ...filteredContent],
    };
    return { ...rawMsg, content: updatedInner };
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: AnthropicApiMessage): SessionMessage {
    // Thinking blocks are already handled by AnthropicTransport.convertFromRawMessage.
    return msg;
  }

  extractFromDelta<T extends Record<string, unknown>>(
    _data: string,
    _delta: RawContentBlockDelta,
    _streamDataExtractor: StreamDataExtractor<T>,
  ): Promise<void> {
    // Not used: AnthropicTransport.generate handles all events natively via for-await.
    return Promise.resolve();
  }

  extractFromResponse(
    _data: 'reasoning_tokens' | 'cache_read_tokens' | 'cache_created_tokens' | string,
    _response: Message,
  ): unknown {
    return undefined;
  }
}
