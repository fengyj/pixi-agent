import { ApiModes, RawMessageType } from '../message';
import {
  ProviderTransport,
  DialectResolver,
  ApiModeResolverRegistry,
  DialectResolverRegistry,
} from './base';
import { OpenRouterApiModeResolver, OpenRouterChatDialectResolver } from './dialects/openrouter';
import {
  DeepSeekApiModeResolver,
  DeepSeekChatDialectResolver,
  DeepSeekAnthropicDialectResolver,
} from './dialects/deepseek';
import { ChatCompletionTransport } from './chat_completion';
import { AnthropicTransport } from './anthropic';

export const Transport = {
  getTransport(
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
        // return new ResponseTransport();
        throw new Error('ResponseTransport is not implemented yet.');
      case ApiModes.ANTHROPIC:
        return new AnthropicTransport(baseUrl, apiKey, dialectResolver);
      case ApiModes.BEDROCK:
        // return new BedrockTransport();
        throw new Error('BedrockTransport is not implemented yet.');
      default:
        // return new OpenAITransport();
        throw new Error('OpenAITransport is not implemented yet.');
    }
  },
  GlobalApiModeResolverRegistry: new ApiModeResolverRegistry()
    .registerResolver(new OpenRouterApiModeResolver())
    .registerResolver(new DeepSeekApiModeResolver()),
  GlobalDialectResolverRegistry: new DialectResolverRegistry()
    .registerResolver(new OpenRouterChatDialectResolver())
    .registerResolver(new DeepSeekChatDialectResolver())
    .registerResolver(new DeepSeekAnthropicDialectResolver()),
};

export type { ModelOptions, StreamCallbacks} from './base';
export { ModelOptionsSchema, ProviderTransport } from './base';
