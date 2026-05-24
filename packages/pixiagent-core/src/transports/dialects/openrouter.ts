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
  ResponseInputItem,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import type {
  Message,
  MessageParam,
  MessageStreamParams,
  RawContentBlockDelta,
} from '@anthropic-ai/sdk/resources/messages/messages';
import { ApiModes, SessionMessage, ThinkingPart, ContentPart } from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions } from '../base';

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
  ChatCompletionMessageParam,
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
    rawMsg: ChatCompletionMessageParam,
    msg?: SessionMessage,
  ): ChatCompletionMessageParam {
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
      return { ...rawMsg, reasoning_details: reasoningDetails } as ChatCompletionMessageParam;
    }

    // Plain text thinking — use the simpler `reasoning` string field.
    const reasoningText = thinkingParts.map((p) => p.content).join('');
    return { ...rawMsg, reasoning: reasoningText } as ChatCompletionMessageParam;
  }

  manipulateMessage(msg: SessionMessage, rawMsg: ChatCompletionMessageParam): SessionMessage {
    if (rawMsg.role !== 'assistant') return msg;

    const raw = rawMsg as ChatCompletionAssistantMessageParam & {
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromDelta(data: string, delta: ChatCompletionChunk.Choice.Delta): any {
    if (data !== 'reasoning') return null;

    // OpenRouter streaming: reasoning can arrive in reasoning_details with either
    // reasoning.text or reasoning.summary blocks depending on upstream provider.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const reasoningDetails = (delta as any).reasoning_details;
    if (Array.isArray(reasoningDetails)) {
      const text = reasoningDetails
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((d: any) => {
          if (d.type === 'reasoning.text') {
            return d.text ?? d.delta ?? '';
          }
          if (d.type === 'reasoning.summary') {
            return d.summary ?? d.text ?? d.delta ?? '';
          }
          return '';
        })
        .join('');
      if (text) return text;
    }

    // Fallback: legacy delta.reasoning (or reasoning_content) field
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (delta as any).reasoning ?? (delta as any).reasoning_content ?? undefined;
  }

  extractFromResponse(
    data: 'reasoning_tokens' | 'cache_read_tokens' | 'cache_created_tokens' | string,
    response: ChatCompletion,
  ): number | undefined {
    if (data === 'cache_created_tokens') {
      return (
        response.usage?.prompt_tokens_details as { cache_write_tokens?: number } | undefined
      )?.cache_write_tokens;
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
  ResponseInputItem,
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

  manipulateRawMessage(rawMsg: ResponseInputItem, _msg?: SessionMessage): ResponseInputItem {
    return rawMsg;
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: ResponseInputItem): SessionMessage {
    return msg;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromDelta(_data: 'reasoning' | string, _delta: ResponseStreamEvent): any {
    return undefined;
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
  MessageParam,
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

  manipulateRawMessage(rawMsg: MessageParam, _msg?: SessionMessage): MessageParam {
    return rawMsg;
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: MessageParam): SessionMessage {
    return msg;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromDelta(_data: 'reasoning' | string, _delta: RawContentBlockDelta): any {
    return undefined;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromResponse(_data: string, _response: Message): any {
    return undefined;
  }
}
