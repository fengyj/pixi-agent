import { z } from 'zod';
import { ApiModes, RawDeltaMessageType, RawLLMParametersType, RawMessageType, SessionMessage, UsageStats } from '../message';
import { ToolDefinitionSchema } from '../tool';

/**
 * The base class for the transport of the provider.
 *
 * The transport provides the abilities to communicate with the providers with the standard API mode sdk.
 * If the provider uses dialect, define a DialectTransport to override.
 */
export abstract class ProviderTransport<TRawMessage> {
  constructor(
    public apiMode: ApiModes,
    public dialectResolver?: DialectResolver<TRawMessage, RawDeltaMessageType, RawLLMParametersType>,
  ) {}

  abstract convertFromRawMessage(rawMsg: TRawMessage): SessionMessage;

  abstract convertToRawMessage(msg: SessionMessage): TRawMessage;

  abstract generate(
    options: ModelOptions,
    messages: TRawMessage[],
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<{ rawMessageId: string; rawMessage: TRawMessage; usage?: UsageStats }>;
}

export type ModelRequestOptions = {
  signal?: AbortSignal | undefined | null;
  timeout?: number;
  maxRetries?: number;
}

export const ModelOptionsSchema = z.object({
  /**
   * The model name.
   */
  model: z.string(),
  /**
   * The base url of the API. If not specified, the default base url will be used.
   */
  baseUrl: z.url().optional(),
  /**
   * The API key for authentication.
   */
  apiKey: z.string().optional(),
  /**
   * The API mode. If not specified, the API mode will be inferred by the baseUrl and model.
   */
  apiMode: z.enum(ApiModes).optional(),
  /**
   * The dialect used by the model. If not specified, the default dialect will be inferred by the baseUrl and model.
   */
  dialect: z.string().optional(),
  /**
   * Output tokens limit for the response.
   * For COMPLETIONS, RESPONSE, the tokens include the reasoning and the final answer.
   * But for ANTHROPIC, the tokens only include the final answer, not include the reasoning.
   */
  maxTokens: z.number().optional(),
  /**
   * Because Anthropic uses system parameter to specify the system prompt,
   * and Response uses instructions parameter to specify the system prompt,
   * For COMPLETIONS, we use this parameter to represent the system message. In other words,
   * the system message won't be converted to an InternalMessage object.
   */
  systemPrompt: z.string().optional(),
  /**
   * Some modelds support the value from 0 to 1, and some others support the value from 0 to 2.
   * We use the value from 0 to 1 for all the models, and convert it to the value range required by the model in the transport layer.
   */
  temperature: z.number().min(0).max(1).optional(),
  /**
   * Response API doesn't support. So when using Response API, the stopSequences will be ignored.
   */
  stopSequences: z.array(z.string()).optional(),
  tools: z.array(ToolDefinitionSchema).optional(),
  /**
   * Force model to use tools.
   * - 'auto': the model can decide when to use tools.
   * - 'none': Response API doesn't support this option, will use 'auto' instead.
   * - 'force': the model will be forced to use tools when generating the response.
   *            Equals to the `required` in COMPLETIONS and RESPONSE APIs, and the `any` in ANTHROPIC API.
   */
  toolChoice: z.union([z.literal('auto'), z.literal('none'), z.literal('force')]).optional(),
  /**
   * To allow or disable parallel tool calls. Defaults to true. Disable it to make sure one tool call each time.
   *
   * for Anthropic API, it's specified in `tool_choice` parameter.
   * ```
   * tool_choice: {
   *   type: "auto",
   *   disable_parallel_tool_use: true
   * }
   * ```
   */
  parallelToolCalls: z.boolean().optional(),
  outputSchema: z
    .object({
      schema: z.record(z.string(), z.any()),
      strict: z.boolean(),
    })
    .optional(),
  /**
   * How much effort the model should put into thinking.
   *
   * For COMPLETIONS API, for the official OpenAI models, it will be ignored since it's not supported.
   * But for some other providers, they may have the models support it.
   * So, need to handle it by the DialectTransport.
   *
   * For RESPONSE API, the value needs to be mapped to "low" | "medium" | "high" | "xhigh"
   * and set to `reasoning.effort`.
   *
   * For ANTHROPIC API, the value needs to be mapped to "low" | "medium" | "high" | "xhigh" | "max"
   * and set to `output_config.effort`.
   */
  thinkEffort: z
    .union([
      z.literal('disable'),
      z.literal('low'),
      z.literal('medium'),
      z.literal('high'),
      z.literal('extreme'),
    ])
    .optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export type ModelOptions = z.infer<typeof ModelOptionsSchema>;

export type StreamCallbacks = {
  onTextChunk?: (delta: string, chunkFlag?: 'begin' | 'end') => Promise<void> | void;
  onText?: (text: string) => Promise<void> | void;
  onThinkingChunk?: (delta: string, chunkFlag?: 'begin' | 'end') => Promise<void> | void;
  onThinking?: (text: string) => Promise<void> | void;
  onToolUse?: (name: string, inputDelta: string) => Promise<void> | void;
  /**
   * Just provide a hook for the error handling. No matter what the hook does,
   * the error will still be thrown to the upper level. So the hook is just for the side effect,
   * e.g. logging the error, or send the error to some monitoring service.
   * @param error
   * @returns
   */
  onError?: (error: Error) => Promise<void> | void;
};

/**
 * Infer the API mode from the model options with the following priority:
 * 1. The apiMode specified in the options.
 * 2. The baseUrl is specified in the options, infer the API mode from the baseUrl and model.
 * 3. If baseUrl is absent, infer the API mode from the model (infer the baseUrl via the model).
 */
export abstract class ApiModeResolver {
  /**
   * Infer the API mode from the model and baseUrl.
   * If the resolver cannot recognize the model and baseUrl, return undefined.
   * @param model
   * @param baseUrl
   */
  abstract getApiMode(model: string, baseUrl?: string): ApiModes | undefined;
  /**
   * Return the official base url for the model. This function shold only be used when the baseUrl
   * is not specified in the options. If the resolver cannot recognize the model, return undefined.
   * @param model the model name has to exactly match the official model name
   */
  abstract getBaseUrl(model: string, apiMode?: ApiModes): string | undefined;
}

/**
 * The resolver is used for manipulating the subtle differences in the ModelOptions and messages
 * between the OpenAI/Anthropic and other providers.
 *
 * Infer the dialect from the model and the baseUrl.
 *
 */
export abstract class DialectResolver<TRawMessage, TRawDelta, TParameters> {
  /**
   * Infer the dialect from the model, api mode and the baseUrl.
   * @param model
   * @param baseUrl
   */
  abstract match(model: string, baseUrl: string): boolean;
  /**
   * Update the model options according to the dialect.
   * @param options
   */
  abstract manipulateOptions(options: ModelOptions, parameters: TParameters): TParameters;
  /**
   * Update the raw message when converting the SessionMessage to the raw message, or when
   * sanitising an existing raw message for the dialect (msg is absent in that case).
   * When msg is absent the implementation should structuredClone rawMsg before mutating it.
   * @param rawMsg the raw message to manipulate.
   * @param msg the SessionMessage that rawMsg was converted from, or undefined when rawMsg
   *            is an already-stored raw message that was not converted from a SessionMessage.
   */
  abstract manipulateRawMessage(rawMsg: TRawMessage, msg?: SessionMessage): TRawMessage;
  /**
   * Update the SessionMessage when converting the raw message to the SessionMessage.
   * @param msg the SessionMessage converted from the raw message
   * via the ProviderTransport's convertFromRawMessage function.
   * @param rawMsg
   */
  abstract manipulateMessage(msg: SessionMessage, rawMsg: TRawMessage): SessionMessage;
  /**
   * Extract the data from the raw message or the raw delta when receiving the stream.
   * @param data the name of the data field.
   * @param delta the raw delta.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  abstract extractFromDelta(data: string, delta: TRawDelta): any;
}

export class ApiModeResolverRegistry {
  private readonly resolvers: ApiModeResolver[] = [];

  public registerResolver(resolver: ApiModeResolver): this {
    this.resolvers.push(resolver);
    return this;
  }

  /**
   * Resolve the API mode, baseUrl.
   * @param model
   * @param baseUrl
   * @param apiMode
   * @returns
   */
  public resolve(
    model: string,
    baseUrl?: string,
    apiMode?: ApiModes,
  ): [string | undefined, ApiModes] | undefined {
    // 1. if baseUrl hasn't been specified, try to infer it via the model name, and the api mode (if it's specified).
    //    if the baseUrl still cannot be determined, leave it as underfined,
    //    fall back to the default provider (OpenAI or Anthropic).
    if (!baseUrl) {
      for (const resolver of this.resolvers) {
        const inferredBaseUrl = resolver.getBaseUrl(model, apiMode);
        if (inferredBaseUrl) {
          baseUrl = inferredBaseUrl;
          break;
        }
      }
    }
    // 2. try to infer the API mode via the model and baseUrl.
    for (const resolver of this.resolvers) {
      const mode = resolver.getApiMode(model, baseUrl);
      if (mode && (!apiMode || mode === apiMode)) {
        return [baseUrl, mode];
      }
    }
    // 3. if the API mode is specified in the options, but cannot be recognized by any resolver,
    //    return the undefined baseUrl and the apiMode, so that can use the default provider.
    //    But if the apiMode is not specified, return undefined because no idea what provider to use.
    return apiMode ? [baseUrl, apiMode] : undefined;
  }
}

export class DialectResolverRegistry {
  private readonly resolvers: DialectResolver<RawMessageType, RawDeltaMessageType, RawLLMParametersType>[] = [];

  public registerResolver(resolver: DialectResolver<RawMessageType, RawDeltaMessageType, RawLLMParametersType>): this {
    this.resolvers.push(resolver);
    return this;
  }

  /**
   * Try to resolve the dialect via the model and the baseUrl.
   * @param model
   * @param baseUrl it must be specified because if it's not specified, the provider must be OpenAI or Anthropic,
   *                no need to specify the dialect.
   * @returns
   */
  public resolveDialect(
    model: string,
    baseUrl: string,
  ): DialectResolver<RawMessageType, RawDeltaMessageType, RawLLMParametersType> | undefined {
    for (const resolver of this.resolvers) {
      if (resolver.match(model, baseUrl)) {
        return resolver;
      }
    }
    return undefined;
  }
}
