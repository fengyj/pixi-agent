import { ApiModes, RawMessageType } from '../message';
import {
  ProviderTransport,
  DialectResolver,
  ApiModeResolverRegistry,
  DialectResolverRegistry,
} from './base';
import {
  OpenRouterApiModeResolver,
  OpenRouterAnthropicDialectResolver,
  OpenRouterChatDialectResolver,
  OpenRouterResponseDialectResolver,
} from './dialects/openrouter';
import {
  OfoxApiModeResolver,
  OfoxAnthropicDialectResolver,
  OfoxChatDialectResolver,
  OfoxResponseDialectResolver,
} from './dialects/ofox';
import {
  DeepSeekApiModeResolver,
  DeepSeekChatDialectResolver,
  DeepSeekAnthropicDialectResolver,
} from './dialects/deepseek';
import { ChatCompletionTransport } from './chat_completion';
import { AnthropicTransport } from './anthropic';
import { ResponseTransport } from './response';

export type { ModelOptions, StreamCallbacks, ProviderTransport } from './base';
export { ModelOptionsSchema } from './base';

export function getTransport(
  apiMode: ApiModes,
  baseUrl?: string,
  apiKey?: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dialectResolver?: DialectResolver<any, any, any, any>,
): ProviderTransport<RawMessageType> {
  switch (apiMode) {
    case ApiModes.COMPLETIONS:
      return new ChatCompletionTransport(baseUrl, apiKey, dialectResolver);
    case ApiModes.RESPONSE:
      return new ResponseTransport(baseUrl, apiKey, dialectResolver);
    case ApiModes.ANTHROPIC:
      return new AnthropicTransport(baseUrl, apiKey, dialectResolver);
    case ApiModes.BEDROCK:
      // return new BedrockTransport();
      throw new Error('BedrockTransport is not implemented yet.');
    default:
      // return new OpenAITransport();
      throw new Error('OpenAITransport is not implemented yet.');
  }
}
export const GlobalApiModeResolverRegistry = new ApiModeResolverRegistry()
  .registerResolver(new OpenRouterApiModeResolver())
  .registerResolver(new OfoxApiModeResolver())
  .registerResolver(new DeepSeekApiModeResolver());
export const GlobalDialectResolverRegistry = new DialectResolverRegistry()
  .registerResolver(new OpenRouterChatDialectResolver())
  .registerResolver(new OpenRouterResponseDialectResolver())
  .registerResolver(new OpenRouterAnthropicDialectResolver())
  .registerResolver(new OfoxChatDialectResolver())
  .registerResolver(new OfoxResponseDialectResolver())
  .registerResolver(new OfoxAnthropicDialectResolver())
  .registerResolver(new DeepSeekChatDialectResolver())
  .registerResolver(new DeepSeekAnthropicDialectResolver());
