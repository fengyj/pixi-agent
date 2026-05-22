import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionStreamParams,
} from 'openai/resources/chat/completions';
import { ApiModes, SessionMessage, ThinkingPart, ContentPart } from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions } from '../base';

export class OpenRouterApiModeResolver extends ApiModeResolver {
  getApiMode(_model: string, baseUrl?: string): ApiModes | undefined {
    if (baseUrl?.toLowerCase() === 'https://openrouter.ai/api/v1') return ApiModes.COMPLETIONS;
    return undefined;
  }

  getBaseUrl(_model: string, _apiMode?: ApiModes): string | undefined {
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
    return baseUrl.toLowerCase() === 'https://openrouter.ai/api/v1';
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

    // OpenRouter streaming: reasoning arrives in delta.reasoning_details[].text
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const reasoningDetails = (delta as any).reasoning_details;
    if (Array.isArray(reasoningDetails)) {
      const text = reasoningDetails
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .filter((d: any) => d.type === 'reasoning.text' && d.text)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        .map((d: any) => d.text as string)
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
  ): any {
    if (data === 'cache_created_tokens') {
      return (response.usage?.prompt_tokens_details as any)?.cache_write_tokens ?? undefined;
    }
    return undefined;
  }
}
