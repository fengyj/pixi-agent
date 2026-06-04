import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionStreamParams,
} from 'openai/resources/chat/completions';
import type {
  Response,
  ResponseCreateParams,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import type {
  Message,
  MessageStreamParams,
  RawContentBlockDelta,
} from '@anthropic-ai/sdk/resources/messages/messages';
import {
  ApiModes,
  AnthropicApiMessage,
  ChatCompletionApiMessage,
  ResponseApiMessage,
  SessionMessage,
  ThinkingPart,
  ContentPart,
} from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions, StreamDataExtractor } from '../base';

const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_RESPONSES_ENDPOINT = 'https://openrouter.ai/api/v1/responses';
const OPENROUTER_ANTHROPIC_MESSAGES_ENDPOINT = 'https://openrouter.ai/api/v1/messages';

function normalizeUrl(url?: string): string | undefined {
  if (!url) return undefined;
  return url.endsWith('/') ? url.slice(0, -1).toLowerCase() : url.toLowerCase();
}

export class OpenRouterApiModeResolver extends ApiModeResolver {
  getApiMode(_model: string, baseUrl?: string): ApiModes | undefined {
    const normalized = normalizeUrl(baseUrl);
    if (normalized === OPENROUTER_ANTHROPIC_MESSAGES_ENDPOINT) return ApiModes.ANTHROPIC;
    if (normalized === OPENROUTER_RESPONSES_ENDPOINT) return ApiModes.RESPONSE;
    if (normalized === OPENROUTER_BASE) return ApiModes.COMPLETIONS;
    return undefined;
  }

  getBaseUrl(_model: string, apiMode?: ApiModes): string | undefined {
    if (apiMode === ApiModes.RESPONSE) return OPENROUTER_RESPONSES_ENDPOINT;
    return undefined;
  }
}

/**
 * Dialect resolver for OpenRouter via the OpenAI Chat Completions API (https://openrouter.ai/api/v1).
 *
 * Handles reasoning tokens using OpenRouter's unified `reasoning` parameter and the
 * `reasoning_details` / `reasoning` response fields.
 *
 * Supported reasoning_details formats:
 *  - `reasoning.text`      (anthropic-claude-v1, openai-responses-v1, …) — text + optional signature
 *  - `reasoning.summary`   (openai-responses-v1, …) — plain summary text, stored without signature.
 *
 * `reasoning.encrypted` blocks are intentionally ignored: they are provider-opaque and meaningless
 * outside the original provider context, so there is nothing useful to preserve.
 */
export class OpenRouterChatDialectResolver implements DialectResolver<
  ChatCompletionApiMessage,
  ChatCompletionChunk.Choice.Delta,
  ChatCompletionStreamParams,
  ChatCompletion
> {
  match(_model: string, baseUrl: string): boolean {
    const normalized = normalizeUrl(baseUrl);
    return normalized === OPENROUTER_BASE;
  }

  manipulateOptions(
    options: ModelOptions,
    parameters: ChatCompletionStreamParams,
  ): ChatCompletionStreamParams {
    if (!options.thinkEffort) return parameters;

    // OpenRouter uses extra_body.reasoning.effort (unified across providers).
    const effortMap: Record<string, string> = {
      disable: 'none',
      low: 'low',
      medium: 'medium',
      high: 'high',
      extreme: 'xhigh',
    };
    const effort = effortMap[options.thinkEffort];
    if (!effort) return parameters;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (parameters as any).extra_body = {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ...((parameters as any).extra_body ?? {}),
      reasoning: { effort },
    };
    return parameters;
  }

  manipulateRawMessage(
    rawMsg: ChatCompletionApiMessage,
    msg?: SessionMessage,
  ): ChatCompletionApiMessage {
    if (rawMsg.role !== 'assistant' || !(msg?.content instanceof Array)) return rawMsg;

    const thinkingParts = msg.content.filter((p): p is ThinkingPart => p.type === 'thinking');
    if (thinkingParts.length === 0) return rawMsg;

    // If any part carries a signature, use reasoning_details so that Anthropic's signed
    // thinking blocks are preserved verbatim.
    const hasStructuredParts = thinkingParts.some((p) => p.signature != null);

    if (hasStructuredParts) {
      const reasoningDetails = thinkingParts.map((p) => ({
        type: 'reasoning.text',
        text: p.content,
        signature: p.signature ?? null,
      }));
      return {
        ...rawMsg,
        content: {
          ...rawMsg.content,
          reasoning_details: reasoningDetails,
        } as unknown as ChatCompletionMessageParam,
      };
    }

    // Plain text thinking — use the simpler `reasoning` string field.
    const reasoningText = thinkingParts.map((p) => p.content).join('');
    return {
      ...rawMsg,
      content: {
        ...rawMsg.content,
        reasoning: reasoningText,
      } as unknown as ChatCompletionMessageParam,
    };
  }

  manipulateMessage(msg: SessionMessage, rawMsg: ChatCompletionApiMessage): SessionMessage {
    const inner = rawMsg.content;
    if (inner.role !== 'assistant') return msg;

    const raw = inner as ChatCompletionAssistantMessageParam & {
      reasoning?: string;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      reasoning_details?: any[];
    };

    const thinkingParts: ThinkingPart[] = [];

    if (raw.reasoning_details && raw.reasoning_details.length > 0) {
      for (const detail of raw.reasoning_details) {
        if (detail.type === 'reasoning.text') {
          thinkingParts.push({
            type: 'thinking',
            content: detail.text ?? '',
            signature: detail.signature ?? undefined,
          });
        } else if (detail.type === 'reasoning.summary') {
          thinkingParts.push({
            type: 'thinking',
            content: detail.summary ?? '',
          });
        }
      }
    } else if (raw.reasoning) {
      thinkingParts.push({ type: 'thinking', content: raw.reasoning });
    }

    if (thinkingParts.length === 0) return msg;

    return { ...msg, content: ContentPart.concat(thinkingParts, msg.content) };
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
      // https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
      if (
        'reasoning_details' in delta &&
        Array.isArray(delta.reasoning_details) &&
        delta.reasoning_details.length > 0 &&
        delta.reasoning_details.every(
          (d) =>
            typeof d === 'object' &&
            typeof d.type === 'string' &&
            (d.type === 'reasoning.text' || d.type === 'reasoning.summary') &&
            (typeof d.text === 'string' || typeof d.summary === 'string'),
        )
      ) {
        for (const detail of delta.reasoning_details) {
          await streamDataExtractor.accumulate(
            {
              key: `reasoning_details_${detail.index}`,
              value: detail,
            },
            (acc, newData) => {
              const message = getMessageObj(acc);
              if (!message) return;
              if (!('reasoning_details' in message) || !Array.isArray(message.reasoning_details)) {
                (
                  message as Record<string, unknown> & { reasoning_details: unknown[] }
                ).reasoning_details = [];
              }
              (message.reasoning_details as unknown[]).push(newData);
            },
            (existing, newData, _acc) => {
              existing.type = existing.type ?? newData.type;
              if ('summary' in newData) {
                existing.summary = `${existing.summary ?? ''}${newData.summary ?? ''}`;
              } else if ('text' in newData) {
                existing.text = `${existing.text ?? ''}${newData.text ?? ''}`;
              }
              existing.id = existing.id ?? newData.id;
              existing.format = existing.format ?? newData.format;
              if ('signature' in newData && typeof newData.signature === 'string') {
                existing.signature = `${existing.signature ?? ''}${newData.signature ?? ''}`;
              }
            },
            (data) => ({ type: 'thinking' as const, content: data.summary ?? (!data.signature || data.signature === '' ? data.text ?? '' : '') }),
          );
        }
      } else if (
        ('reasoning' in delta &&
          typeof delta.reasoning === 'string' &&
          delta.reasoning.length > 0) ||
        ('reasoning_content' in delta &&
          typeof delta.reasoning_content === 'string' &&
          delta.reasoning_content.length > 0)
      ) {
        const propName = 'reasoning' in delta ? 'reasoning' : 'reasoning_content';
        await streamDataExtractor.accumulate(
          { key: propName, value: (delta as Record<string, unknown>)[propName] as string },
          (acc, newData) => {
            const message = getMessageObj(acc);
            if (!message) return;
            if (!(propName in message) || typeof message[propName] !== 'string') {
              (message as Record<string, unknown> & { [key: string]: string })[propName] = '';
            }
            (message as Record<string, unknown> & { [key: string]: string })[propName] += newData;
          },
          null,
          (data) => ({ type: 'thinking' as const, content: data }),
        );
      }
    }
  }

  extractFromResponse(
    data: 'reasoning_tokens' | 'cache_read_tokens' | 'cache_created_tokens' | string,
    response: ChatCompletion,
  ): number | undefined {
    if (data === 'cache_created_tokens') {
      return (response.usage?.prompt_tokens_details as { cache_write_tokens?: number } | undefined)
        ?.cache_write_tokens;
    }
    return undefined;
  }
}

/**
 * Dialect resolver for OpenRouter via the OpenAI Responses API (https://openrouter.ai/api/v1).
 *
 * OpenRouter's responses endpoint is largely OpenAI-compatible. This resolver keeps
 * the shape pass-through and only applies OpenRouter-specific option normalization.
 */
export class OpenRouterResponseDialectResolver implements DialectResolver<
  ResponseApiMessage,
  ResponseStreamEvent,
  ResponseCreateParams,
  Response
> {
  match(_model: string, baseUrl: string): boolean {
    const normalized = normalizeUrl(baseUrl);
    return normalized === OPENROUTER_RESPONSES_ENDPOINT;
  }

  manipulateOptions(options: ModelOptions, parameters: ResponseCreateParams): ResponseCreateParams {
    if (!options.thinkEffort) return parameters;

    // Keep effort mapping aligned with OpenRouter Responses `reasoning.effort`.
    const effortMap: Record<string, 'none' | 'low' | 'medium' | 'high' | 'xhigh'> = {
      disable: 'none',
      low: 'low',
      medium: 'medium',
      high: 'high',
      extreme: 'xhigh',
    };
    const effort = effortMap[options.thinkEffort];
    if (!effort) return parameters;

    return {
      ...parameters,
      reasoning: {
        ...(parameters.reasoning ?? {}),
        effort,
      },
    };
  }

  manipulateRawMessage(rawMsg: ResponseApiMessage, _msg?: SessionMessage): ResponseApiMessage {
    return rawMsg;
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: ResponseApiMessage): SessionMessage {
    return msg;
  }

  extractFromDelta<T extends Record<string, unknown>>(
    _data: 'reasoning' | string,
    _delta: ResponseStreamEvent,
    _streamDataExtractor: StreamDataExtractor<T>,
  ): Promise<void> {
    // https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    // todo: implement reasoning data from delta
    return Promise.resolve();
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromResponse(_data: string, _response: Response): any {
    return undefined;
  }
}

/**
 * Dialect resolver for OpenRouter via Anthropic Messages API endpoint
 * (https://openrouter.ai/api/v1/messages).
 *
 * OpenRouter's Anthropic endpoint is largely Anthropic-compatible, so we only
 * keep a light pass-through dialect hook for transport unification.
 */
export class OpenRouterAnthropicDialectResolver implements DialectResolver<
  AnthropicApiMessage,
  RawContentBlockDelta,
  MessageStreamParams,
  Message
> {
  match(_model: string, baseUrl: string): boolean {
    return normalizeUrl(baseUrl) === OPENROUTER_ANTHROPIC_MESSAGES_ENDPOINT;
  }

  manipulateOptions(_options: ModelOptions, parameters: MessageStreamParams): MessageStreamParams {
    return parameters;
  }

  manipulateRawMessage(rawMsg: AnthropicApiMessage, _msg?: SessionMessage): AnthropicApiMessage {
    return rawMsg;
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: AnthropicApiMessage): SessionMessage {
    return msg;
  }

  extractFromDelta<T extends Record<string, unknown>>(
    _data: 'reasoning' | string,
    _delta: RawContentBlockDelta,
    _streamDataExtractor: StreamDataExtractor<T>,
  ): Promise<void> {
    // https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    // todo: implement reasoning data from delta
    return Promise.resolve();
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromResponse(_data: string, _response: Message): any {
    return undefined;
  }
}
