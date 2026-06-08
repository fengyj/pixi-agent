import type {  RawContentBlockDelta } from '@anthropic-ai/sdk/resources/messages';
import type { Message, MessageCreateParamsStreaming } from '@anthropic-ai/sdk/resources/messages/messages';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
} from 'openai/resources/chat/completions/completions';
import type {
  Response,
  ResponseCreateParamsStreaming,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import { z } from 'zod';

import {
  ApiModes,
  ContentPart,
  RawMessageType,
  SessionMessage,
} from '../message';
import { ToolDefinitionSchema } from '../tools/tool';


type RawDeltaMessageType =
  | ChatCompletionChunk.Choice.Delta
  | ResponseStreamEvent
  | RawContentBlockDelta;

type RawLLMParametersType =
  | MessageCreateParamsStreaming
  | ChatCompletionCreateParamsStreaming
  | ResponseCreateParamsStreaming;

type RawResponseType = ChatCompletion | Response | Message;

/**
 * The base class for the transport of the provider.
 *
 * The transport provides the abilities to communicate with the providers with the standard API mode sdk.
 * If the provider uses dialect, define a DialectTransport to override.
 */
export abstract class ProviderTransport<TRawMessage> {
  constructor(
    public readonly apiMode: ApiModes,
    public readonly dialectResolver?: DialectResolver<
      TRawMessage,
      RawDeltaMessageType,
      RawLLMParametersType,
      RawResponseType
    >,
  ) {}

  /**
   * Convert the raw message to SessionMessage.
   *
   * @param rawMsg
   */
  abstract convertFromRawMessage(rawMsg: TRawMessage): SessionMessage;

  /**
   * Convert the SessionMessage to raw message(s).
   * When the SessionMessage is tool message, and when it's converted to ChatCompletion/Response
   * message, because the API's tool result message only supports one tool result,
   * so the SessionMessage will be converted to multiple raw messages, one for each tool result,
   * and if the message contains any other content besides the tool result,
   * it will be converted to one more raw user message.
   * @param msg
   */
  abstract convertToRawMessage(msg: SessionMessage): TRawMessage | TRawMessage[];

  abstract generate(
    options: ModelOptions,
    messages: TRawMessage[],
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<Omit<TRawMessage, 'messageId'>>;
}

export interface ModelRequestOptions {
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
export abstract class DialectResolver<TRawMessage, TRawDelta, TParameters, TRawResponse> {
  /**
   * Infer the dialect from the model and the baseUrl.
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
   * @param streamDataExtractor the extracted data will be handled by streamDataExtractor
   *                            to put the delta data to the accumulatedData.
   *                            And if the callbacks parameter is provided,
   *                            emit the chunk data via the callbacks.
   */
  abstract extractFromDelta<T extends object>(
    data: 'reasoning' | string,
    delta: TRawDelta,
    streamDataExtractor: StreamDataExtractor<T>,
  ): Promise<void>;
  /**
   * Extract the data from the raw message when receiving the final response.
   * @param data the name of the data field.
   * @param response the raw response.
   */
  abstract extractFromResponse(
    data:
      | 'reasoning_tokens'
      | 'cache_read_tokens'
      | 'cache_created_tokens'
      | 'stop_reason'
      | string,
    response: TRawResponse, // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): any;
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
  private readonly resolvers: DialectResolver<
    RawMessageType,
    RawDeltaMessageType,
    RawLLMParametersType,
    RawResponseType
  >[] = [];

  public registerResolver(
    resolver: DialectResolver<
      RawMessageType,
      RawDeltaMessageType,
      RawLLMParametersType,
      RawResponseType
    >,
  ): this {
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
  ):
    | DialectResolver<RawMessageType, RawDeltaMessageType, RawLLMParametersType, RawResponseType>
    | undefined {
    for (const resolver of this.resolvers) {
      if (resolver.match(model, baseUrl)) {
        return resolver;
      }
    }
    return undefined;
  }
}

export interface StreamCallbacks {
  onChunk: (chunk: {
    contentPartIndex: number;
    contentPartChunk: ContentPart;
    chunkIndex: number;
  }) => Promise<void>;

  onFinish: (msg: SessionMessage) => Promise<void>;
}

/**
 * The StreamDataExtractor is used for accumulating the data from the stream and 
 * emit the chunk data via the callbacks.
 * 
 * The class cannot handle the logic of when a content part is completed, because
 * a delta data could contain more than one content part data, we don't know the orders of them,
 * so there is no way to determine when a content part is completed. For example,
 * the content part 7 is contained in the previous delta data, and it also is in the current one.
 * But there is another content part 8 in it, and we process it before the content part 7.
 * If need to resolve this issue, have to check the content part if exists in the next delta data.
 * This will make the logic much more complicated.
 */
export class StreamDataExtractor<T extends object> {
  public readonly accumulatedData: T;
  /**
   * The contentPartIndex means the nth content part emitted in the stream.
   * The chunkIndex means the nth chunk of the content part emitted.
   */
  private readonly blocks: Map<
    string,
    { contentPartIndex: number; chunkIndex: number; data: unknown }
  > = new Map();
  private blockCount: number = 0;
  private contentPartCount: number = 0;

  constructor(
    initialData: T,
    private readonly callBacks?: StreamCallbacks,
  ) {
    this.accumulatedData = initialData;
  }

  /**
   *
   * @param delta the key is used for merging existing block. when it's absent, means no merging.
   * @param append define how to append the new block to the accumulated object.
   * @param merge define how to merge the new delta with the existing block when the key is the same.
   *              if it's null, means no merging, just append as a new block (the key of the delta has to be undefined).
   * @param toContentPart define how to convert the delta data to a ContentPart.
   *                      If the result is null, the chunk won't be emitted via the callbacks.
   */
  async accumulate<P>(
    delta: { key?: string; value: P },
    append: (accumulated: T, newData: P) => void,
    merge: ((existing: P, newData: P, accumulated: T) => void) | null,
    toContentPart: (data: P) => ContentPart | null,
  ): Promise<void> {
    const singleChunkBlock = !delta.key;
    if (singleChunkBlock && merge !== null) {
      throw new Error('When the delta has no key, the merge function has to be null.');
    }
    if (!singleChunkBlock && !merge) {
      throw new Error('When the delta has a key, the merge function cannot be null.');
    }
    const key = delta.key ?? `single_chunk_block_${this.blockCount}`;
    const value = delta.value;
    const isNewBlock = !this.blocks.has(key);
    const block = this.blocks.get(key) ?? {
      contentPartIndex: -1,
      chunkIndex: -1,
      data: value,
    };

    if (isNewBlock) {
      this.blockCount++;
      this.blocks.set(key, block);
      append(this.accumulatedData, value);
    } else if (!singleChunkBlock) {
      merge!(block.data as P, value, this.accumulatedData);
    }
    if (this.callBacks) {
      const isNewContentPart = block.chunkIndex === -1;
      const contentPart = toContentPart(value);
      if (contentPart) {
        if (isNewContentPart) {
          block.contentPartIndex = this.contentPartCount++;
        }
        block.chunkIndex++;
        await this.callBacks.onChunk({
          contentPartIndex: block.contentPartIndex,
          contentPartChunk: contentPart,
          chunkIndex: block.chunkIndex,
        });
      }
    }
  }
}
