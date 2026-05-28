import type { MessageParam, RawContentBlockDelta } from '@anthropic-ai/sdk/resources/messages';
import type { Message, MessageStreamParams } from '@anthropic-ai/sdk/resources/messages/messages';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionStreamParams,
} from 'openai/resources/chat/completions/completions';
import type {
  Response,
  ResponseCreateParams,
  ResponseInputItem,
  ResponseStreamEvent,
} from 'openai/resources/responses/responses';
import { z } from 'zod';

export enum ApiModes {
  COMPLETIONS = 'completions',
  RESPONSE = 'response',
  ANTHROPIC = 'anthropic',
  BEDROCK = 'bedrock', // maybe this should be a dialect
}

const RoleTypeSchema = z.enum(['assistant', 'user', 'tool']);
export type RoleType = z.infer<typeof RoleTypeSchema>;

export interface UsageStats {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cacheReadTokens?: number;
  cacheCreatedTokens?: number;
  reasoningTokens?: number;
  inputTokenDetails?: {
    [name: string]: number;
  };
  outputTokenDetails?: {
    [name: string]: number;
  };
  // todo: add price field
}

function mergeTokenDetails(
  a?: UsageStats['inputTokenDetails'],
  b?: UsageStats['inputTokenDetails'],
): UsageStats['inputTokenDetails'] | undefined {
  if (!a && !b) {
    return undefined;
  }

  const merged: Record<string, number> = {};

  if (a) {
    for (const [key, value] of Object.entries(a)) {
      merged[key] = value;
    }
  }

  if (b) {
    for (const [key, value] of Object.entries(b)) {
      merged[key] = (merged[key] ?? 0) + value;
    }
  }

  return merged;
}

export const UsageStats = {
  empty: (): UsageStats => ({
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cacheReadTokens: 0,
    cacheCreatedTokens: 0,
    reasoningTokens: 0,
  }),
  sum: (a: UsageStats, b?: UsageStats): UsageStats => {
    if (!b) return a;

    return {
      inputTokens: a.inputTokens + b.inputTokens,
      outputTokens: a.outputTokens + b.outputTokens,
      totalTokens: a.totalTokens + b.totalTokens,
      cacheReadTokens: (a.cacheReadTokens ?? 0) + (b.cacheReadTokens ?? 0),
      cacheCreatedTokens: (a.cacheCreatedTokens ?? 0) + (b.cacheCreatedTokens ?? 0),
      reasoningTokens: (a.reasoningTokens ?? 0) + (b.reasoningTokens ?? 0),
      inputTokenDetails: mergeTokenDetails(a.inputTokenDetails, b.inputTokenDetails),
      outputTokenDetails: mergeTokenDetails(a.outputTokenDetails, b.outputTokenDetails),
    };
  },
  isEmpty: (usage: UsageStats): boolean => {
    return (
      usage.inputTokens === 0 &&
      usage.outputTokens === 0 &&
      usage.totalTokens === 0 &&
      (usage.cacheReadTokens ?? 0) === 0 &&
      (usage.cacheCreatedTokens ?? 0) === 0 &&
      (usage.reasoningTokens ?? 0) === 0 &&
      (usage.inputTokenDetails === undefined ||
        Object.values(usage.inputTokenDetails).every((v) => v === 0)) &&
      (usage.outputTokenDetails === undefined ||
        Object.values(usage.outputTokenDetails).every((v) => v === 0))
    );
  },
};

/**
 * The media source used in the message content.
 *
 * The `file_id` sourceType is used for the media that has been uploaded to the server,
 * and can be referenced by the fileId.
 * The `expireAt` field is optional, and is used to indicate the expiration time of the file.
 * After the expiration time, the file will be deleted from the server
 * and can no longer be accessed by the fileId.
 * @todo we can support base64 and url source types first. They are easier to handle the expiration and provider change issues. For the file_id source type, we need to handle the expiration and provider change issues, which is more complex. So we can support it in the future after we have a better understanding of the requirements and constraints.
 * @todo for the source which type is 'file_id', handle the case when the file is expired
 * or the provider is changed, needs to reupload the file, and replace the fileId in all references in the whole history.
 */
const MediaSourceSchema = z.union([
  z.object({
    sourceType: z.literal('base64'),
    /**
     * COMPLETIONS API:
     * - audio: 'audio/wav' | 'audio/mp3' <-> 'wav' | 'mp3'
     */
    mimeType: z.string(),
    data: z.string(),
    /**
     * The file name of the media, which is optional.
     * It can be used to indicate the original file name of the media.
     */
    fileName: z.string().optional(),
    /**
     * The URI of the media, which is optional.
     * It can be used to indicate the location of the media.
     */
    uri: z.string().optional(),
  }),
  z.object({
    sourceType: z.literal('url'),
    url: z.string(),
    /**
     * The file name of the media, which is optional.
     * It can be used to indicate the original file name of the media.
     */
    fileName: z.string().optional(),
    expireAt: z.number().optional(),
    /**
     * The URI of the media, which is optional.
     * It can be used to indicate the original location of the media.
     */
    uri: z.string().optional(),
  }),
  z.object({
    sourceType: z.literal('file_id'),
    fileId: z.string(),
    expireAt: z.number().optional(),
    /**
     * The file name of the media, which is optional.
     * It can be used to indicate the original file name of the media.
     */
    fileName: z.string().optional(),
    /**
     * The URI of the media, which is optional.
     * It can be used to indicate the original location of the media.
     */
    uri: z.string().optional(),
  }),
]);

export type MediaSource = z.infer<typeof MediaSourceSchema>;

const CitationWebLocationSchema = z.object({
  url: z.string(),
  type: z.literal('web_location'),
  citedText: z.string(),
  title: z.string().optional(),
  startIndex: z.number().optional(),
  endIndex: z.number().optional(),
  /** For the fields not very common in the original citation/annotation data */
  extra: z.record(z.string(), z.unknown()).optional(),
});

export type CitationWebLocation = z.infer<typeof CitationWebLocationSchema>;

const CitationFileLocationSchema = z.object({
  fileName: z.string(),
  type: z.literal('file_location'),
  citedText: z.string(),
  title: z.string().optional(),
  /** can be start_page_number/start_block_index/start_index/etc. */
  startIndex: z.number().optional(),
  /** can be end_page_number/end_block_index/end_index/etc. */
  endIndex: z.number().optional(),
  /** Used for identifying the citation/annotation type from the raw message */
  rawCitationType: z.string(),
  /** For the fields not very common in the original citation/annotation data */
  extra: z.record(z.string(), z.unknown()).optional(),
});

export type CitationFileLocation = z.infer<typeof CitationFileLocationSchema>;

const CitationOthersLocationSchema = z.object({
  source: z.string(),
  type: z.literal('others_location'),
  citedText: z.string(),
  title: z.string().optional(),
  startIndex: z.number().optional(),
  endIndex: z.number().optional(),
  /** Used for identifying the citation/annotation type from the raw message */
  rawCitationType: z.string(),
  /** For the fields not very common in the original citation/annotation data */
  extra: z.record(z.string(), z.unknown()).optional(),
});

export type CitationOthersLocation = z.infer<typeof CitationOthersLocationSchema>;

const CitationSchema = z.union([
  CitationWebLocationSchema,
  CitationFileLocationSchema,
  CitationOthersLocationSchema,
]);

export type Citation = z.infer<typeof CitationSchema>;

const TextPartSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
  citations: z.array(CitationSchema).optional(),
});

export type TextPart = z.infer<typeof TextPartSchema>;

const ThinkingPartSchema = z.object({
  type: z.literal('thinking'),
  content: z.string(),
  signature: z.string().optional(),
});

export type ThinkingPart = z.infer<typeof ThinkingPartSchema>;

const RefusalPartSchema = z.object({
  type: z.literal('refusal'),
  reason: z.string(),
});

export type RefusalPart = z.infer<typeof RefusalPartSchema>;

const ImagePartSchema = z.object({
  type: z.literal('image'),
  image: MediaSourceSchema,
});

export type ImagePart = z.infer<typeof ImagePartSchema>;

const DocumentPartSchema = z.object({
  type: z.literal('document'),
  document: MediaSourceSchema,
});

export type DocumentPart = z.infer<typeof DocumentPartSchema>;

const AudioPartSchema = z.object({
  type: z.literal('audio'),
  audio: MediaSourceSchema,
});

export type AudioPart = z.infer<typeof AudioPartSchema>;

const VideoPartSchema = z.object({
  type: z.literal('video'),
  video: MediaSourceSchema,
});

export type VideoPart = z.infer<typeof VideoPartSchema>;

const ToolCallPartSchema = z.object({
  type: z.literal('tool_call'),
  id: z.string(),
  name: z.string(),
  /**
   * The JSON string of the arguments.
   *
   * The type of the arguments (`input`) in Anthropic message is `Record<string, unknown>`,
   * needs to convert the data to JSON.
   */
  arguments: z.string(),
});

export type ToolCallPart = z.infer<typeof ToolCallPartSchema>;

const ToolResultPartSchema = z.object({
  type: z.literal('tool_result'),
  id: z.string(),
  name: z.string().optional(),
  /**
   * The JSON string of the result.
   * The Anthropic supports the image as the result of the tool call.
   */
  result: z.union([
    z.string(),
    z.array(
      z.union([
        TextPartSchema,
        ImagePartSchema,
        DocumentPartSchema,
        AudioPartSchema,
        VideoPartSchema,
      ]),
    ),
  ]),
  isError: z.boolean().optional(),
});

export type ToolResultPart = z.infer<typeof ToolResultPartSchema>;

const ContentPartSchema = z.union([
  TextPartSchema,
  ThinkingPartSchema,
  RefusalPartSchema,
  ImagePartSchema,
  DocumentPartSchema,
  AudioPartSchema,
  VideoPartSchema,
  ToolCallPartSchema,
  ToolResultPartSchema,
]);

export type ContentPart = z.infer<typeof ContentPartSchema>;

/**
 * Normalize content into ContentPart[] so callers can process mixed string/parts
 * message content in a uniform way.
 */
function toContentParts(content?: string | Array<ContentPart>): Array<ContentPart> {
  if (content === undefined) {
    return [];
  }
  if (typeof content === 'string') {
    return [{ type: 'text', text: content } as TextPart];
  }
  return content;
}

/**
 * Concatenate two content fragments represented as text or parts.
 */
function concatContentParts(
  part1?: string | Array<ContentPart>,
  part2?: string | Array<ContentPart>,
): Array<ContentPart> {
  return [...toContentParts(part1), ...toContentParts(part2)];
}

function getContentDigest(
  content?: string | Array<ContentPart>,
): string | Array<ContentPart> | undefined {
  if (content === undefined) return undefined;

  const maxLength = 20;
  const headLength = 10;
  const tailLength = 5;

  const digestString = (value: string): string => {
    if (value.length <= maxLength) {
      return value;
    }
    return `${value.slice(0, headLength)}...${value.slice(-tailLength)}`;
  };

  const digestValue = (value: unknown): unknown => {
    if (typeof value === 'string') {
      return digestString(value);
    }

    if (Array.isArray(value)) {
      return value.map((item) => digestValue(item));
    }

    if (value instanceof Date) {
      return new Date(value.getTime());
    }

    if (value && typeof value === 'object') {
      const digestedObject: Record<string, unknown> = {};
      for (const [key, fieldValue] of Object.entries(value)) {
        digestedObject[key] = digestValue(fieldValue);
      }
      return digestedObject;
    }

    return value;
  };

  return digestValue(content) as string | Array<ContentPart>;
}

function createProviderToolCallArguments(
  provider: string,
  rawType: string,
  rawRequest: unknown,
): string {
  return JSON.stringify({ provider, rawType, rawRequest });
}

export const ContentPart = {
  toParts: toContentParts,
  concat: concatContentParts,
  digest: getContentDigest,
  createProviderToolCallArguments: createProviderToolCallArguments,
};

/**
 * An API neutral message format. Used for UI, and as an intermediate data for the conversion
 * between the different API modes or dialects.
 */
export const SessionMessageSchema = z.object({
  messageId: z.string(),
  type: z.literal('session_message'),
  role: RoleTypeSchema,
  content: z.union([z.string(), z.array(ContentPartSchema)]),
  /**
   * The name of the role, which is optional.
   * It can be used to indicate the name of the assistant or user.
   */
  name: z.string().optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type SessionMessage = z.infer<typeof SessionMessageSchema>;

/**
 * The message in the OpenAI's response API, is quite different from the others.
 * It isn't message based. So, have to define a type for it.
 */
export type ResponseApiMessage = {
  messageId: string;
  type: 'response_api_message';
  role: RoleType;
  content: Array<ResponseInputItem>;
  metadata?: Record<string, unknown>;
};

export type ChatCompletionApiMessage = {
  messageId: string;
  type: 'chat_completion_api_message';
  role: RoleType;
  content: ChatCompletionMessageParam;
  metadata?: Record<string, unknown>;
};

export type AnthropicApiMessage = {
  messageId: string;
  type: 'anthropic_api_message';
  role: RoleType;
  content: MessageParam;
  metadata?: Record<string, unknown>;
};

export type RawMessageType = ChatCompletionApiMessage | ResponseApiMessage | AnthropicApiMessage;

export type RawDeltaMessageType =
  | ChatCompletionChunk.Choice.Delta
  | ResponseStreamEvent
  | RawContentBlockDelta;

export type RawLLMParametersType =
  | MessageStreamParams
  | ChatCompletionStreamParams
  | ResponseCreateParams;

export type RawResponseType = ChatCompletion | Response | Message;

/**
 * The structure used for persistence.
 *
 * We store the raw message from the API directly. If the API mode or the dialect isn't changed in the next turn,
 * the raw message can be used directly as the input. This can help us avoid the information loss during the conversion between different API modes or dialects.
 * If the API mode or the dialect is changed in the next turn, the old messages will be converted to the SessionMessage and
 * then converted to the new raw format of the new model. Because users rarely change the model during the conversation,
 * so I think it's better then saving the messages in SessionMessage format.
 */
export interface InternalMessage {
  /**
   * uuid v6, internal usage only.
   */
  internalMessageId: string;
  /**
   * The model used of the message.
   */
  model: string;
  /**
   * The API mode used of the message.
   */
  apiMode: ApiModes;
  /**
   * The baseUrl of the API used for this message.
   */
  baseUrl?: string;
  /**
   * The raw message object is used as the input of the next turn of the conversation.
   * If the apiMode of the message is the same as the next turn,
   * the rawMessage will be used directly as the input of the next turn.
   * If the apiMode of the message is different from the next turn,
   * the rawMessage will be transformed into the input format of the next turn.
   * For example, if the current message is from openai chat completions
   * and the next message is from anthropic, the rawMessage will be transformed from
   * ChatCompletionMessageParam to MessageParam before being used as the input of the next turn.
   * This allows us to keep the original message format for each API mode,
   * while still being able to use them interchangeably in the conversation.
   *
   * For the user input and tool messages, use the SessionMessage format, because
   * the format may contain the information or data that is not supported by the API,
   * to avoid those information or data being lost, so we use the SessionMessage format here.
   *
   * When converting to other message formats, here are somethings need to be considered:
   * - The Anthropic message could contain text and tool calls at the same time,
   *   and OpenAI's usually doesn't.
   * - The user role and assistant role in Anthropic messages has to be occurred alternatively,
   *   like user -> assistant -> user -> assistant, while OpenAI's messages doesn't have such requirement.
   * - The RedactedThinkBlock in the Anthropic message contains unreadable content for users, that means the content in it
   *   is also useless for other API modes, so we can just ignore it when converting to other message formats.
   * - ServerToolUseBlock needs to be converted to tool calls.
   * - The tool result blocks need to be converted to tool messages in OpenAI format.
   * - The ContainerUploadBlock also doesn't need to be converted.
   * - The system message in OpenAI format shouldn't be converted to Anthropic message,
   *   because Anthropic doesn't have system role, extract the content and use it for the system parameter.
   *
   * So, the conversion is not always one-to-one. For example, when converting an Anthropic message
   * to OpenAI format, if the message contains both text and tool calls,
   * we may need to split it into two messages in OpenAI format,
   * one for the text and one for the tool call.
   * And when converting from OpenAI format to Anthropic format,
   * if we have two consecutive messages with the same role,
   * we may need to merge them into one message in Anthropic format.
   *
   * Here are some more details for other LLM providers:
   * - Deepseek
   *   - reasoning_content: the thinking content will be returned in this field, the same level as the content field.
   *     And this field shouldn't be included in the input of next turn.
   * - Qwen
   *   - reasoning_content: the thinking mode only can be triggered when the content is simple text, not array.
   */
  rawMessage: RawMessageType | SessionMessage;
  /**
   * The role of the message, which can be 'assistant', 'user', or 'tool'.
   */
  role: RoleType;
  /**
   * The message happened before this one. Excepts the first message, others should have this value.
   */
  previousMessageId?: string;
  usage?: UsageStats;
  createdAt: string;
  completedAt?: string;
}
