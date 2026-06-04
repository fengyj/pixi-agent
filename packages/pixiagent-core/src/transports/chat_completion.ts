import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ModelResponse,
  ProviderTransport,
  StreamCallbacks,
  StreamDataExtractor,
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
  ChatCompletionMessage,
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
  CitationWebLocation,
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
          arguments: ContentPart.createProviderToolCallArguments(
            'openai_chat',
            toolCall.type,
            toolCall,
          ),
        } as ToolCallPart);
      });
    }
    const annotations = this.convertChatCompletionAnnotationsToCitations(
      (
        msg as ChatCompletionAssistantMessageParam & {
          annotations?: Array<ChatCompletionMessage.Annotation>;
        }
      ).annotations,
    );

    if (typeof msg.content === 'string') {
      if (parts.length === 0 && annotations.length === 0) {
        return {
          messageId: rawMsg.messageId,
          type: 'session_message',
          role: 'assistant',
          content: msg.content,
          name: msg.name,
          metadata: rawMsg.metadata,
        } as SessionMessage;
      }
      parts.push({
        type: 'text',
        text: msg.content,
        citations: annotations.length > 0 ? annotations : undefined,
      } as TextPart);
    } else if (msg.content instanceof Array) {
      msg.content.forEach((contentPart) => {
        if (contentPart.type === 'text') {
          parts.push(this.convertFromChatCompletionTextPart(contentPart));
        } else if (contentPart.type === 'refusal') {
          parts.push(this.convertFromChatCompletionRefusalPart(contentPart));
        }
      });
      if (annotations.length > 0) {
        this.attachCitationsToTextParts(parts, annotations);
      }
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

  private convertChatCompletionAnnotationsToCitations(
    annotations?: Array<ChatCompletionMessage.Annotation>,
  ): Array<CitationWebLocation> {
    if (!annotations || annotations.length === 0) {
      return [];
    }

    return annotations
      .filter(
        (annotation): annotation is ChatCompletionMessage.Annotation =>
          annotation.type === 'url_citation',
      )
      .map((annotation) => ({
        type: 'web_location',
        url: annotation.url_citation.url,
        citedText: '',
        title: annotation.url_citation.title,
        startIndex: annotation.url_citation.start_index,
        endIndex: annotation.url_citation.end_index,
        extra: { rawCitationType: 'url_citation' },
      }));
  }

  private attachCitationsToTextParts(
    parts: ContentPart[],
    citations: Array<CitationWebLocation>,
  ): ContentPart[] {
    if (citations.length === 0) {
      return parts;
    }

    const firstText = parts.find((part) => part.type === 'text') as TextPart | undefined;
    if (firstText) {
      firstText.citations = [...(firstText.citations ?? []), ...citations];
      return parts;
    }

    return [
      {
        type: 'text',
        text: '',
        citations,
      } as TextPart,
      ...parts,
    ];
  }

  private convertCitationToChatCompletionAnnotation(
    citation: CitationWebLocation,
  ): ChatCompletionMessage.Annotation {
    return {
      type: 'url_citation',
      url_citation: {
        url: citation.url,
        title: citation.title ?? '',
        start_index: citation.startIndex ?? 0,
        end_index: citation.endIndex ?? 0,
      },
    };
  }

  private getChatCompletionAnnotationsFromSessionMessage(
    msg: SessionMessage,
  ): Array<ChatCompletionMessage.Annotation> {
    if (!Array.isArray(msg.content)) {
      return [];
    }

    return msg.content.flatMap((part) => {
      if (part.type !== 'text' || !part.citations) {
        return [];
      }
      return part.citations
        .filter((citation): citation is CitationWebLocation => citation.type === 'web_location')
        .map((citation) => this.convertCitationToChatCompletionAnnotation(citation));
    });
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

    const annotations = this.getChatCompletionAnnotationsFromSessionMessage(msg);

    return {
      role: 'assistant',
      content:
        messageLevelRefusal || (content instanceof Array && content.length === 0) ? null : content,
      tool_calls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
      audio: null, // todo: may need to handle this in the future.
      name: msg.name,
      refusal: messageLevelRefusal,
      ...(annotations.length > 0 ? { annotations } : {}),
    } as ChatCompletionAssistantMessageParam & {
      annotations?: Array<ChatCompletionMessage.Annotation>;
    };
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
      try {
        const resultObj = JSON.parse(toolResult.result ?? '{}');
        if (
          Array.isArray(resultObj) &&
          resultObj.every(
            (item) =>
              typeof item === 'object' &&
              'type' in item &&
              item.type === 'text' &&
              'text' in item &&
              typeof item.text === 'string',
          )
        ) {
          return {
            role: 'tool',
            tool_call_id: toolResult.id,
            content: resultObj.map((item) => ({ type: 'text', text: item.text })),
          };
        } else {
          return {
            role: 'tool',
            tool_call_id: toolResult.id,
            content: toolResult.result ?? JSON.stringify(null),
          };
        }
      } catch {
        return {
          role: 'tool',
          tool_call_id: toolResult.id,
          content: toolResult.result ?? JSON.stringify(null),
        };
      }
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
  ): ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>> {
    const responseMessage = response.choices[0]?.message;
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

  async generate(
    options: ModelOptions,
    messages: Array<ChatCompletionApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ChatCompletionApiMessage, 'messageId'>>> {
    const params = this.getChatCompletionStreamParams(options, messages);
    try {
      const stream = await this.client.chat.completions.create(
        { ...params, stream: true },
        this.getStreamRequestOptions(requestOptions),
      );
      const streamDataExtractor = new StreamDataExtractor(
        {
          id: '',
          model: undefined as ChatCompletion['model'] | undefined,
          created: Date.now() / 1000,
          usage: undefined as ChatCompletion['usage'] | undefined,
          choices: [{
            index: 0,
            finish_reason: 'stop' as ChatCompletion.Choice['finish_reason'],
            logprobs: null,
            message: {
              role: 'assistant',
              content: null as Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal> | null,
              tool_calls: undefined as Array<ChatCompletionMessageFunctionToolCall> | undefined,
              audio: null,
            },
          }],
        },
        callbacks,
      );

      for await (const chunk of stream) {
        if (chunk.choices.length === 0) continue;
        if (chunk.usage) {
          streamDataExtractor.accumulatedData.usage = chunk.usage;
        }
        if (chunk.id) {
          streamDataExtractor.accumulatedData.id = chunk.id;
        }
        streamDataExtractor.accumulatedData.model = chunk.model;
        streamDataExtractor.accumulatedData.created = chunk.created;
        if (chunk.choices.length > 1) {
          continue; // Not support multiple choices.
        }
        const choice = chunk.choices[0];
        if (choice.finish_reason) {
          streamDataExtractor.accumulatedData.choices[0].finish_reason = choice.finish_reason;
        }

        if(this.dialectResolver) {
          await this.dialectResolver.extractFromDelta(
            'reasoning',
            choice.delta,
            streamDataExtractor
          );
        }

        if (choice.delta.content && choice.delta.content.length > 0) {
          await streamDataExtractor.accumulate(
            {
              value: { type: 'text', text: choice.delta.content } as ChatCompletionContentPartText,
            },
            (accumulated, newData) => {
              if(accumulated.choices[0].message.content === null) {
                accumulated.choices[0].message.content = [];
              }
              accumulated.choices[0].message.content.push(newData);
            },
            null,
            (newData) => ({ type: 'text', text: newData.text }),
          );
        }
        if (choice.delta.refusal && choice.delta.refusal.length > 0) {
          await streamDataExtractor.accumulate(
            {
              value: {
                type: 'refusal',
                refusal: choice.delta.refusal,
              } as ChatCompletionContentPartRefusal,
            },
            (accumulated, newData) => {
              if (accumulated.choices[0].message.content === null) {
                accumulated.choices[0].message.content = [];
              }
              accumulated.choices[0].message.content.push(newData);
            },
            null,
            (newData) => ({ type: 'refusal', reason: newData.refusal }),
          );
        }
        if (choice.delta.tool_calls) {
          for (const tc of choice.delta.tool_calls) {
            await streamDataExtractor.accumulate(
              {
                key: `tool_call_${tc.index}`,
                value: {
                  type: 'function',
                  id: tc.id ?? '',
                  function: tc.function ?? { name: '', arguments: '' },
                } as ChatCompletionMessageFunctionToolCall,
              },
              (accumulated, newData) => {
                if (accumulated.choices[0].message.tool_calls === undefined) {
                  accumulated.choices[0].message.tool_calls = [];
                }
                accumulated.choices[0].message.tool_calls.push(newData);
              },
              (existing, newData) => {
                existing.id += newData.id ?? '';
                existing.function.name += newData.function?.name ?? '';
                existing.function.arguments += newData.function?.arguments ?? '';
              },
              (newData) => ({
                type: 'tool_call',
                id: newData.id ?? '',
                name: newData.function?.name ?? '',
                arguments: newData.function?.arguments ?? '',
              }),
            );
          }
        }
      }
      const response = {
        ...streamDataExtractor.accumulatedData,
        object: 'chat.completion',
      } as ChatCompletion;

      return this.buildModelResponse(response);
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
