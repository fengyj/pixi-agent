import type {
  ChatCompletion,
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
import { ApiModes, SessionMessage } from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions } from '../base';

const OFOX_OPENAI_BASES = ['https://api.ofox.ai/v1', 'https://api.ofox.io/v1'];
const OFOX_CHAT_ENDPOINTS = [
  'https://api.ofox.ai/v1/chat/completions',
  'https://api.ofox.io/v1/chat/completions',
];
const OFOX_RESPONSES_ENDPOINTS = [
  'https://api.ofox.ai/v1/responses',
  'https://api.ofox.io/v1/responses',
];
const OFOX_ANTHROPIC_BASES = ['https://api.ofox.ai/anthropic', 'https://api.ofox.io/anthropic'];
const OFOX_ANTHROPIC_MESSAGES_ENDPOINTS = [
  'https://api.ofox.ai/anthropic/v1/messages',
  'https://api.ofox.io/anthropic/v1/messages',
];

function isIn(url: string | undefined, candidates: string[]): boolean {
  if (!url) return false;
  return candidates.includes(url);
}

function normalizeUrl(url?: string): string | undefined {
  if (!url) return undefined;
  return url.endsWith('/') ? url.slice(0, -1).toLowerCase() : url.toLowerCase();
}

export class OfoxApiModeResolver extends ApiModeResolver {
  getApiMode(_model: string, baseUrl?: string): ApiModes | undefined {
    const normalized = normalizeUrl(baseUrl);
    if (!normalized) return undefined;

    if (isIn(normalized, OFOX_OPENAI_BASES) || isIn(normalized, OFOX_CHAT_ENDPOINTS)) {
      return ApiModes.COMPLETIONS;
    }
    if (isIn(normalized, OFOX_RESPONSES_ENDPOINTS)) {
      return ApiModes.RESPONSE;
    }
    if (isIn(normalized, OFOX_ANTHROPIC_BASES) || isIn(normalized, OFOX_ANTHROPIC_MESSAGES_ENDPOINTS)) {
      return ApiModes.ANTHROPIC;
    }

    return undefined;
  }

  getBaseUrl(_model: string, _apiMode?: ApiModes): string | undefined {
    // OFOX supports multiple protocols and endpoint styles; do not auto-infer to avoid ambiguity.
    return undefined;
  }
}

export class OfoxChatDialectResolver implements DialectResolver<
  ChatCompletionMessageParam,
  ChatCompletionChunk.Choice.Delta,
  ChatCompletionStreamParams,
  ChatCompletion
> {
  match(_model: string, baseUrl: string): boolean {
    const normalized = normalizeUrl(baseUrl);
    return isIn(normalized, OFOX_OPENAI_BASES) || isIn(normalized, OFOX_CHAT_ENDPOINTS);
  }

  manipulateOptions(
    _options: ModelOptions,
    parameters: ChatCompletionStreamParams,
  ): ChatCompletionStreamParams {
    return parameters;
  }

  manipulateRawMessage(
    rawMsg: ChatCompletionMessageParam,
    _msg?: SessionMessage,
  ): ChatCompletionMessageParam {
    return rawMsg;
  }

  manipulateMessage(msg: SessionMessage, _rawMsg: ChatCompletionMessageParam): SessionMessage {
    return msg;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromDelta(_data: 'reasoning' | string, _delta: ChatCompletionChunk.Choice.Delta): any {
    return undefined;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  extractFromResponse(_data: string, _response: ChatCompletion): any {
    return undefined;
  }
}

export class OfoxResponseDialectResolver implements DialectResolver<
  ResponseInputItem,
  ResponseStreamEvent,
  ResponseCreateParams,
  Response
> {
  match(_model: string, baseUrl: string): boolean {
    return isIn(normalizeUrl(baseUrl), OFOX_RESPONSES_ENDPOINTS);
  }

  manipulateOptions(options: ModelOptions, parameters: ResponseCreateParams): ResponseCreateParams {
    if (!options.thinkEffort) return parameters;

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

export class OfoxAnthropicDialectResolver implements DialectResolver<
  MessageParam,
  RawContentBlockDelta,
  MessageStreamParams,
  Message
> {
  match(_model: string, baseUrl: string): boolean {
    const normalized = normalizeUrl(baseUrl);
    return isIn(normalized, OFOX_ANTHROPIC_BASES) || isIn(normalized, OFOX_ANTHROPIC_MESSAGES_ENDPOINTS);
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
