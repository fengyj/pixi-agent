import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ModelResponse,
  ProviderTransport,
  StreamCallbacks,
} from './base';
import type {
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionContentPartText,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionContentPartRefusal,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartInputAudio,
  ChatCompletionContentPart,
  ChatCompletionFunctionMessageParam,
  ChatCompletionStreamParams,
  ChatCompletionChunk,
  ChatCompletionReasoningEffort,
  ChatCompletion,
} from 'openai/resources/chat/completions';
import {
  ApiModes,
  ChatCompletionApiMessage,
  SessionMessage,
  ToolCallPart,
  TextPart,
  RefusalPart,
  AudioPart,
  ImagePart,
  DocumentPart,
  ToolResultPart,
  ContentPart,
} from '../message';
import {
  AgentInterruptedError,
  InvalidMessageError,
  ModelRequestTimeoutError,
} from '../errors/types';
import { isLikelyAbortError, isLikelyTimeoutError } from '../errors/guards';
import { OpenAI } from 'openai/client';

export class ChatCompletionTransport extends ProviderTransport<ChatCompletionApiMessage> {
  readonly client: OpenAI;
  private static readonly OFFICIAL_BASE_URL = 'https://api.openai.com/v1';
  private readonly configuredBaseUrl?: string;

  private static normalizeBaseUrl(baseUrl?: string): string | undefined {
    if (!baseUrl) return baseUrl;
    const normalized = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    if (normalized.toLowerCase().endsWith('/chat/completions')) {
      return normalized.slice(0, -'/chat/completions'.length);
    }
    return normalized;
  }

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      ChatCompletionApiMessage,
      ChatCompletionChunk.Choice.Delta,
      ChatCompletionStreamParams,
      ChatCompletion
    >,
  ) {
    super(ApiModes.COMPLETIONS, dialectResolver);
    this.configuredBaseUrl = ChatCompletionTransport.normalizeBaseUrl(baseUrl);
    this.client = new OpenAI({
      baseURL: this.configuredBaseUrl,
      apiKey: apiKey,
    });
  }

  getApiMode(): ApiModes {
    return ApiModes.COMPLETIONS;
  }

  private getFromAssistantMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionAssistantMessageParam;
    const parts = [] as ContentPart[];

    if (msg.refusal) {
      parts.push({
        type: 'refusal',
        reason: msg.refusal,
      } as RefusalPart);
    }
    if (msg.audio) {
      parts.push({
        type: 'audio',
        audio: {
          sourceType: 'file_id',
          fileId: msg.audio.id,
        },
      } as AudioPart);
    }
    if (msg.function_call) {
      parts.push({
        type: 'tool_call',
        id: '',
        name: msg.function_call.name,
        arguments: msg.function_call.arguments,
      } as ToolCallPart);
    }
    if (msg.tool_calls) {
      msg.tool_calls.forEach((toolCall) => {
        if (toolCall.type === 'function') {
          parts.push({
            type: 'tool_call',
            id: toolCall.id,
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          } as ToolCallPart);
          return;
        }

        // For custom/non-function tool call variants, preserve raw provider data
        parts.push({
          type: 'tool_call',
          id: toolCall.id,
          name:
            typeof (toolCall as { custom?: { name?: string } }).custom?.name === 'string'
              ? (toolCall as { custom: { name: string } }).custom.name
              : toolCall.id,
          arguments: ContentPart.createProviderToolCallArguments('openai_chat', toolCall.type, toolCall),
        } as ToolCallPart);
      });
    }
    if (typeof msg.content === 'string') {
      if (parts.length === 0) {
        return {
          messageId: rawMsg.messageId,
          type: 'session_message',
          role: 'assistant',
          content: msg.content,
          name: msg.name,
          metadata: rawMsg.metadata,
        } as SessionMessage;
      } else {
        parts.push({
          type: 'text',
          text: msg.content,
        } as TextPart);
      }
    } else if (msg.content instanceof Array) {
      msg.content.forEach((contentPart) => {
        if (contentPart.type === 'text') {
          parts.push(this.convertFromChatCompletionTextPart(contentPart));
        } else if (contentPart.type === 'refusal') {
          parts.push(this.convertFromChatCompletionRefusalPart(contentPart));
        }
      });
    }

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'assistant',
      content: parts,
      name: msg.name,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromUserMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionUserMessageParam;
    if (typeof msg.content === 'string') {
      return {
        messageId: rawMsg.messageId,
        type: 'session_message',
        role: 'user',
        content: msg.content,
        name: msg.name,
        metadata: rawMsg.metadata,
      } as SessionMessage;
    }

    const parts = [] as ContentPart[];
    msg.content.forEach((contentPart) => {
      if (contentPart.type === 'text') {
        parts.push(this.convertFromChatCompletionTextPart(contentPart));
      } else if (contentPart.type === 'image_url') {
        parts.push(this.convertFromChatCompletionImagePart(contentPart));
      } else if (contentPart.type === 'input_audio') {
        parts.push(this.convertFromChatCompletionAudioPart(contentPart));
      } else if (contentPart.type === 'file') {
        const filePart = this.convertFromChatCompletionFilePart(contentPart);
        if (filePart) {
          parts.push(filePart);
        }
      }
    });

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'user',
      content: parts,
      name: msg.name,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromToolMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionToolMessageParam;
    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw new InvalidMessageError('Tool message must have content');
    }

    const toolResultParts = [] as ToolResultPart[];
    if (typeof msg.content === 'string') {
      toolResultParts.push({
        type: 'tool_result',
        id: msg.tool_call_id,
        result: msg.content,
      } as ToolResultPart);
    } else if (msg.content instanceof Array) {
      msg.content.forEach((contentPart) => {
        if (contentPart.type === 'text') {
          toolResultParts.push({
            type: 'tool_result',
            id: msg.tool_call_id,
            result: contentPart.text,
          } as ToolResultPart);
        } 
      });
    }

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'tool',
      content: toolResultParts,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromFunctionMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionFunctionMessageParam;
    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: '',
          name: msg.name,
          arguments: msg.content,
        } as ToolCallPart,
      ],
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private convertFromChatCompletionTextPart(part: ChatCompletionContentPartText): TextPart {
    return { type: 'text', text: part.text };
  }

  private convertFromChatCompletionRefusalPart(
    part: ChatCompletionContentPartRefusal,
  ): RefusalPart {
    return { type: 'refusal', reason: part.refusal };
  }

  private convertFromChatCompletionImagePart(part: ChatCompletionContentPartImage): ImagePart {
    const isBase64 = part.image_url.url.startsWith('data:image/');
    if (isBase64) {
      const mediaType = part.image_url.url.split(';')[0].split(':')[1];
      const base64Data = part.image_url.url.split(',')[1];
      return {
        type: 'image',
        image: {
          sourceType: 'base64',
          data: base64Data,
          mimeType: mediaType,
        },
      };
    }
    return {
      type: 'image',
      image: {
        sourceType: 'url',
        url: part.image_url.url,
      },
    };
  }

  private convertFromChatCompletionAudioPart(part: ChatCompletionContentPartInputAudio): AudioPart {
    return {
      type: 'audio',
      audio: {
        sourceType: 'base64',
        data: part.input_audio.data,
        mimeType: `audio/${part.input_audio.format}`,
      },
    };
  }

  private convertFromChatCompletionFilePart(
    part: ChatCompletionContentPart.File,
  ): DocumentPart | null {
    if (part.file.file_data) {
      return {
        type: 'document',
        document: {
          sourceType: 'base64',
          mimeType: 'application/octet-stream', // todo: check the file name if possible to determine the mime type
          data: part.file.file_data,
          fileName: part.file.filename ?? undefined,
        },
      };
    }

    if (part.file.file_id) {
      return {
        type: 'document',
        document: {
          sourceType: 'file_id',
          fileId: part.file.file_id,
          fileName: part.file.filename ?? undefined,
        },
      };
    }

    return null;
  }

  private convertToChatCompletionTextPart(part: TextPart): ChatCompletionContentPartText {
    return { type: 'text', text: part.text };
  }

  private convertToChatCompletionRefusalPart(part: RefusalPart): ChatCompletionContentPartRefusal {
    return { type: 'refusal', refusal: part.reason };
  }

  private convertToChatCompletionImagePart(part: ImagePart): ChatCompletionContentPartImage | null {
    switch (part.image.sourceType) {
      case 'url':
        return { type: 'image_url', image_url: { url: part.image.url } };
      case 'base64':
        return {
          type: 'image_url',
          image_url: { url: `data:${part.image.mimeType};base64,${part.image.data}` },
        };
      default:
        return null;
    }
  }

  private convertToChatCompletionAudioPart(
    part: AudioPart,
  ): ChatCompletionContentPartInputAudio | null {
    if (part.audio.sourceType !== 'base64') {
      return null;
    }
    const format = part.audio.mimeType.replace('audio/', '');
    return {
      type: 'input_audio',
      input_audio: {
        data: part.audio.data,
        format: format as 'wav' | 'mp3',
      },
    };
  }

  private convertToChatCompletionFilePart(part: DocumentPart): ChatCompletionContentPart.File {
    return {
      type: 'file',
      file: {
        file_data: part.document.sourceType === 'base64' ? part.document.data : undefined,
        file_id: part.document.sourceType === 'file_id' ? part.document.fileId : undefined,
        filename: part.document.fileName ?? undefined,
      },
    };
  }

  private convertToChatCompletionAssistantContentPart(
    part: ContentPart,
  ): ChatCompletionContentPartText | ChatCompletionContentPartRefusal | null {
    if (part.type === 'text') {
      return this.convertToChatCompletionTextPart(part as TextPart);
    }
    if (part.type === 'refusal') {
      return this.convertToChatCompletionRefusalPart(part as RefusalPart);
    }
    return null;
  }

  private convertToChatCompletionUserContentPart(
    part: ContentPart,
  ): ChatCompletionContentPart | null {
    switch (part.type) {
      case 'text':
        return this.convertToChatCompletionTextPart(part as TextPart);
      case 'image':
        return this.convertToChatCompletionImagePart(part as ImagePart);
      case 'audio':
        return this.convertToChatCompletionAudioPart(part as AudioPart);
      case 'document':
        return this.convertToChatCompletionFilePart(part as DocumentPart);
      default:
        return null;
    }
  }

  convertFromRawMessage(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const message = (() => {
      switch (rawMsg.content.role) {
        case 'assistant':
          return this.getFromAssistantMessageParam(rawMsg);
        case 'user':
          return this.getFromUserMessageParam(rawMsg);
        case 'tool':
          return this.getFromToolMessageParam(rawMsg);
        case 'function':
          return this.getFromFunctionMessageParam(rawMsg);
        default:
          throw new InvalidMessageError(`Unsupported message role: ${rawMsg.content.role}`);
      }
    })();
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(message, rawMsg) : message;
  }

  private getAssistantMessageParam(msg: SessionMessage): ChatCompletionAssistantMessageParam {
    if (msg.role !== 'assistant') {
      throw new InvalidMessageError(`Message role must be assistant, but got ${msg.role}`);
    }

    const toolCalls =
      msg.content instanceof Array
        ? msg.content
            .filter((part) => part.type === 'tool_call')
            .map((part) => part as ToolCallPart)
            .map(
              (part) =>
                ({
                  type: 'function',
                  id: part.id,
                  function: {
                    name: part.name,
                    arguments: part.arguments,
                  },
                }) as ChatCompletionMessageFunctionToolCall,
            )
        : undefined;

    const content =
      msg.content instanceof Array
        ? msg.content
            .map((part) => this.convertToChatCompletionAssistantContentPart(part))
            .filter(
              (part): part is ChatCompletionContentPartText | ChatCompletionContentPartRefusal =>
                part !== null,
            )
        : (msg.content ?? null);
        
    let messageLevelRefusal: string | undefined = undefined;
    if (content instanceof Array && content.length === 1 && content[0].type === 'refusal') {
      messageLevelRefusal = content[0].refusal;
    }

    return {
      role: 'assistant',
      content:
        (messageLevelRefusal || (content instanceof Array && content.length === 0)) ? null : content,
      tool_calls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
      audio: null, // todo: may need to handle this in the future.
      name: msg.name,
      refusal: messageLevelRefusal,
    } as ChatCompletionAssistantMessageParam;
  }

  private getUserMessageParam(msg: SessionMessage): ChatCompletionUserMessageParam {
    if (msg.role !== 'user') {
      throw new InvalidMessageError(`Message role must be user, but got ${msg.role}`);
    }
    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw new InvalidMessageError('User message must have content');
    }

    if (typeof msg.content === 'string') {
      return {
        role: 'user',
        content: msg.content,
        name: msg.name,
      } as ChatCompletionUserMessageParam;
    }

    const content = msg.content
      .map((part) => this.convertToChatCompletionUserContentPart(part))
      .filter((part): part is ChatCompletionContentPart => part !== null);

    return {
      role: 'user',
      content: content.length === 0 ? null : content,
      name: msg.name,
    } as ChatCompletionUserMessageParam;
  }

  private getToolMessageParam(
    msg: SessionMessage,
  ): ChatCompletionMessageParam | ChatCompletionMessageParam[] {
    if (msg.role !== 'tool') {
      throw new InvalidMessageError(`Message role must be tool, but got ${msg.role}`);
    }

    if (!msg.content || typeof msg.content === 'string' || msg.content.length === 0) {
      throw new InvalidMessageError(
        'Tool message content must be a non-empty array of content parts.',
      );
    }

    const toolResults = msg.content.filter(
      (part) => part.type === 'tool_result',
    ) as ToolResultPart[];
    const otherParts = msg.content.filter((part) => part.type !== 'tool_result');

    if (toolResults.length === 0) {
      throw new InvalidMessageError(
        'Tool message content must have at least one tool result part.',
      );
    }

    const rawMessages: ChatCompletionMessageParam[] = toolResults.map((toolResult) => {
      const content =
        typeof toolResult.result === 'string'
          ? toolResult.result
          : toolResult.result.map((part) => {
              if (part.type === 'text') {
                return this.convertToChatCompletionTextPart(part as TextPart);
              } else {
                return this.convertToChatCompletionTextPart({
                  type: 'text',
                  text: JSON.stringify(part),
                } as TextPart);
              }
            });

      return {
        role: 'tool',
        tool_call_id: toolResult.id,
        content,
      };
    });

    if (otherParts.length > 0) {
      rawMessages.push(
        this.getUserMessageParam({
          ...msg,
          role: 'user',
          content: otherParts,
        } as SessionMessage),
      );
    }

    return rawMessages.length === 1 ? rawMessages[0] : rawMessages;
  }

  convertToRawMessage(msg: SessionMessage): ChatCompletionApiMessage | ChatCompletionApiMessage[] {
    const raw = (() => {
      switch (msg.role) {
        case 'assistant':
          return this.getAssistantMessageParam(msg);
        case 'user':
          return this.getUserMessageParam(msg);
        case 'tool':
          return this.getToolMessageParam(msg);
        default:
          throw new InvalidMessageError(`Unsupported message role: ${msg.role}`);
      }
    })();

    const wrap = (inner: ChatCompletionMessageParam): ChatCompletionApiMessage => {
      const wrapped: ChatCompletionApiMessage = {
        messageId: msg.messageId,
        type: 'chat_completion_api_message',
        role:
          inner.role === 'function'
            ? 'tool'
            : inner.role === 'developer' || inner.role === 'system' // actually, this shouldn't happen
              ? 'user'
              : inner.role,
        content: inner,
        metadata: msg.metadata,
      };
      return this.dialectResolver
        ? this.dialectResolver.manipulateRawMessage(wrapped, msg)
        : wrapped;
    };

    if (Array.isArray(raw)) {
      const wrapped = raw.map(wrap);
      return wrapped.length === 1 ? wrapped[0] : wrapped;
    }
    return wrap(raw);
  }

  private isOfficialService(): boolean {
    const normalized = ChatCompletionTransport.normalizeBaseUrl(
      this.configuredBaseUrl,
    )?.toLowerCase();
    return !normalized || normalized === ChatCompletionTransport.OFFICIAL_BASE_URL;
  }

  private getStreamRequestOptions(requestOptions?: ModelRequestOptions): {
    signal?: AbortSignal;
    timeout?: number;
  } {
    const streamRequestOptions: {
      signal?: AbortSignal;
      timeout?: number;
    } = {};
    if (requestOptions?.signal) {
      streamRequestOptions.signal = requestOptions.signal;
    }
    if (typeof requestOptions?.timeout === 'number') {
      streamRequestOptions.timeout = requestOptions.timeout;
    }
    return streamRequestOptions;
  }

  private buildModelResponse(
    response: ChatCompletion,
    responseMessage: ChatCompletionMessageParam,
  ): ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>> {
    const wrappedMessage: Omit<ChatCompletionApiMessage, 'messageId'> = {
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: responseMessage,
      metadata: {
        pixiagent_response_id: response.id,
        pixiagent_response_finish_reason: response.choices[0]?.finish_reason,
      },
    };
    return {
      responseId: response.id,
      responseMessage: wrappedMessage,
      responseModel: response.model,
      stopReason:
        this.dialectResolver?.extractFromResponse('stop_reason', response) ??
        this.getStopReason(response.choices[0]?.finish_reason),
      refusal:
        responseMessage.role === 'assistant' && 'refusal' in responseMessage
          ? ((responseMessage as ChatCompletionAssistantMessageParam).refusal ?? undefined)
          : undefined,
      usage: response.usage
        ? {
            inputTokens: response.usage.prompt_tokens,
            outputTokens: response.usage.completion_tokens,
            totalTokens: response.usage.total_tokens,
            reasoningTokens:
              this.dialectResolver?.extractFromResponse('reasoning_tokens', response) ??
              response.usage.completion_tokens_details?.reasoning_tokens ??
              undefined,
            cacheReadTokens:
              this.dialectResolver?.extractFromResponse('cache_read_tokens', response) ??
              response.usage.prompt_tokens_details?.cached_tokens ??
              undefined,
            cacheCreatedTokens:
              this.dialectResolver?.extractFromResponse('cache_created_tokens', response) ??
              undefined,
            inputTokenDetails: response.usage.prompt_tokens_details
              ? { ...response.usage.prompt_tokens_details }
              : undefined,
            outputTokenDetails: response.usage.completion_tokens_details
              ? { ...response.usage.completion_tokens_details }
              : undefined,
          }
        : undefined,
    } as ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>>;
  }

  private async generateWithOfficialStream(
    params: ChatCompletionStreamParams,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>>> {
    let textChunkStarted = false;
    let thinkingChunkStarted = false;
    let thinkingText = '';

    const response = await this.client.chat.completions
      .stream(params, this.getStreamRequestOptions(requestOptions))
      .on('chunk', (chunk) => {
        const choice = chunk.choices[0];
        if (!choice) return;

        const delta = choice.delta;

        const reasoningDelta: string | null | undefined = this.dialectResolver?.extractFromDelta(
          'reasoning',
          delta,
        );
        if (reasoningDelta) {
          thinkingText += reasoningDelta;
          if (!thinkingChunkStarted) {
            thinkingChunkStarted = true;
            callbacks?.onThinkingChunk?.(reasoningDelta, 'begin');
          } else {
            callbacks?.onThinkingChunk?.(reasoningDelta);
          }
        }

        const textDelta: string | null | undefined = delta.content;
        if (textDelta) {
          if (thinkingChunkStarted) {
            thinkingChunkStarted = false;
            callbacks?.onThinkingChunk?.('', 'end');
            callbacks?.onThinking?.(thinkingText);
          }

          if (!textChunkStarted) {
            textChunkStarted = true;
            callbacks?.onTextChunk?.(textDelta, 'begin');
          } else {
            callbacks?.onTextChunk?.(textDelta);
          }
        }

        if (choice.finish_reason) {
          if (textChunkStarted) {
            textChunkStarted = false;
            callbacks?.onTextChunk?.('', 'end');
          }
          if (thinkingChunkStarted) {
            thinkingChunkStarted = false;
            callbacks?.onThinkingChunk?.('', 'end');
            callbacks?.onThinking?.(thinkingText);
          }
        }
      })
      .on('content', (content) => {
        callbacks?.onText?.(content);
      })
      .on('finalFunctionToolCall', (toolCall) => {
        callbacks?.onToolUse?.(toolCall.name, toolCall.arguments);
      })
      .on('error', (error) => {
        callbacks?.onError?.(error);
      })
      .on('abort', (error) => {
        callbacks?.onError?.(error);
      })
      .finalChatCompletion();

    return this.buildModelResponse(
      response,
      response.choices[0].message as ChatCompletionMessageParam,
    );
  }

  private async generateWithThirdPartyCreate(
    params: ChatCompletionStreamParams,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>>> {
    let textChunkStarted = false;
    let thinkingChunkStarted = false;
    let thinkingText = '';
    let finalFinishReason: string | undefined;
    let responseId = '';
    let responseModel = params.model;
    let responseUsage: ChatCompletion['usage'] | undefined;
    let refusalText = '';
    let assistantText = '';
    const toolCalls = new Map<number, ChatCompletionMessageFunctionToolCall>();

    const stream = await this.client.chat.completions.create(
      { ...params, stream: true },
      this.getStreamRequestOptions(requestOptions),
    );

    for await (const chunk of stream) {
      responseId = chunk.id;
      responseModel = chunk.model;
      responseUsage = chunk.usage ?? responseUsage;

      const choice = chunk.choices[0];
      if (!choice) continue;

      const delta = choice.delta;

      const reasoningDelta: string | null | undefined = this.dialectResolver?.extractFromDelta(
        'reasoning',
        delta,
      );
      if (reasoningDelta) {
        thinkingText += reasoningDelta;
        if (!thinkingChunkStarted) {
          thinkingChunkStarted = true;
          callbacks?.onThinkingChunk?.(reasoningDelta, 'begin');
        } else {
          callbacks?.onThinkingChunk?.(reasoningDelta);
        }
      }

      if (delta.content) {
        if (thinkingChunkStarted) {
          thinkingChunkStarted = false;
          callbacks?.onThinkingChunk?.('', 'end');
          callbacks?.onThinking?.(thinkingText);
        }
        assistantText += delta.content;
        if (!textChunkStarted) {
          textChunkStarted = true;
          callbacks?.onTextChunk?.(delta.content, 'begin');
        } else {
          callbacks?.onTextChunk?.(delta.content);
        }
      }

      if (delta.refusal) {
        refusalText += delta.refusal;
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          const existing =
            toolCalls.get(idx) ??
            ({
              type: 'function',
              id: tc.id ?? '',
              function: {
                name: tc.function?.name ?? '',
                arguments: tc.function?.arguments ?? '',
              },
            } as ChatCompletionMessageFunctionToolCall);

          if (tc.id) {
            existing.id = tc.id;
          }
          if (tc.function?.name) {
            existing.function.name = tc.function.name;
          }
          if (tc.function?.arguments) {
            existing.function.arguments += tc.function.arguments;
          }
          toolCalls.set(idx, existing);
        }
      }

      if (choice.finish_reason) {
        finalFinishReason = choice.finish_reason;
      }
    }

    if (textChunkStarted) {
      callbacks?.onTextChunk?.('', 'end');
    }
    if (thinkingChunkStarted) {
      callbacks?.onThinkingChunk?.('', 'end');
      callbacks?.onThinking?.(thinkingText);
    }
    if (assistantText) {
      callbacks?.onText?.(assistantText);
    }

    const message: ChatCompletionAssistantMessageParam = {
      role: 'assistant',
      content: assistantText || null,
      refusal: refusalText || undefined,
      tool_calls:
        toolCalls.size > 0
          ? [...toolCalls.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v)
          : undefined,
      audio: null,
    };

    const syntheticResponse = {
      id: responseId || `chatcmpl_${Date.now().toString(36)}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: responseModel,
      choices: [
        {
          index: 0,
          finish_reason: finalFinishReason ?? 'stop',
          logprobs: null,
          message,
        },
      ],
      usage: responseUsage,
    } as unknown as ChatCompletion;

    return this.buildModelResponse(syntheticResponse, message);
  }

  async generate(
    options: ModelOptions,
    messages: Array<ChatCompletionApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>>> {
    const params = this.getChatCompletionStreamParams(options, messages);
    try {
      return this.isOfficialService()
        ? await this.generateWithOfficialStream(params, callbacks, requestOptions)
        : await this.generateWithThirdPartyCreate(params, callbacks, requestOptions);
    } catch (error) {
      throw this.wrapRequestError(error, requestOptions);
    }
  }

  private wrapRequestError(error: unknown, requestOptions?: ModelRequestOptions): unknown {
    const signal = requestOptions?.signal;
    if (signal?.aborted) {
      const reason = signal.reason;
      if (reason instanceof AgentInterruptedError) {
        return reason;
      }
      if (typeof reason === 'string' && reason.length > 0) {
        return new AgentInterruptedError(reason);
      }
      if (reason instanceof Error && reason.message) {
        return new AgentInterruptedError(reason.message);
      }
      return new AgentInterruptedError();
    }
    if (isLikelyAbortError(error)) {
      return new AgentInterruptedError();
    }
    if (!isLikelyTimeoutError(error)) {
      return error;
    }
    return new ModelRequestTimeoutError(this.client.baseURL, requestOptions?.timeout, error);
  }

  private getStopReason(finishReason: string): string {
    switch (finishReason) {
      case 'stop':
        return 'stop';
      case 'tool_calls':
      case 'function_call':
        return 'tool_call';
      case 'length':
        return 'max_tokens';
      case 'content_filter':
        return 'refusal';
      default:
        return finishReason;
    }
  }

  /**
   * Determines the reasoning_effort value based on model capabilities and thinkEffort option.
   *
   * - O-series models (o1, o3, o4-mini, …) and gpt-5.x models support reasoning_effort.
   * - gpt-5.<digit> models (e.g. gpt-5.1, gpt-5.4-nano) support the 'none' value.
   * - O-series and gpt-5-<name> models (e.g. gpt-5-pro) do NOT support 'none'.
   * - Non-reasoning models → undefined (field omitted from params).
   */
  private getReasoningEffort(
    model: string,
    thinkEffort?: ModelOptions['thinkEffort'],
  ): ChatCompletionReasoningEffort | undefined {
    const baseName = model.includes('/') ? model.split('/').pop()! : model;
    const isOSeries = /^o\d/i.test(baseName);
    const isGpt5x = /^gpt-5/i.test(baseName);
    if (!isOSeries && !isGpt5x) return undefined;

    // gpt-5.<digit> models support 'none'; o-series and gpt-5-<name> do not
    const supportsNone = /^gpt-5\.\d/i.test(baseName);
    switch (thinkEffort) {
      case 'disable':
        return supportsNone ? 'none' : null;
      case 'low':
        return 'low';
      case 'medium':
        return 'medium';
      case 'high':
        return 'high';
      case 'extreme':
        return 'xhigh';
      default:
        return null; // use model default
    }
  }

  private getChatCompletionStreamParams(
    options: ModelOptions,
    messages: Array<ChatCompletionApiMessage>,
  ): ChatCompletionStreamParams {
    const inputs: ChatCompletionMessageParam[] = [];
    if (options.systemPrompt) {
      inputs.push({
        role: 'developer',
        content: options.systemPrompt,
      });
    }
    inputs.push(...messages.map((m) => m.content));

    const params: ChatCompletionStreamParams = {
      model: options.model,
      messages: inputs,
      max_completion_tokens: options.maxTokens,
      temperature: options.temperature,
      metadata: options.metadata,
      reasoning_effort: this.getReasoningEffort(options.model, options.thinkEffort),
      store: false,
      stream: true,
      tools: options.tools?.map((t) => ({
        type: 'function',
        function: t,
      })),
    };

    return this.dialectResolver
      ? (this.dialectResolver.manipulateOptions(options, params) as ChatCompletionStreamParams)
      : params;
  }
}
