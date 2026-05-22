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

export class ChatCompletionTransport extends ProviderTransport<ChatCompletionMessageParam> {
  readonly client: OpenAI;

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      ChatCompletionMessageParam,
      ChatCompletionChunk.Choice.Delta,
      ChatCompletionStreamParams,
      ChatCompletion
    >,
  ) {
    super(ApiModes.COMPLETIONS, dialectResolver);
    this.client = new OpenAI({
      baseURL: baseUrl,
      apiKey: apiKey,
    });
  }

  getApiMode(): ApiModes {
    return ApiModes.COMPLETIONS;
  }

  private getFromAssistantMessageParam(msg: ChatCompletionAssistantMessageParam): SessionMessage {
    if (msg.refusal) {
      return {
        type: 'session_message',
        role: 'assistant',
        content: undefined,
        refusal: msg.refusal,
        name: msg.name,
      } as SessionMessage;
    }

    const parts = [] as ContentPart[];
    // msg.audio is skipped
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
        }
        // skip the custom tool
      });
    }

    if (typeof msg.content === 'string') {
      if (parts.length === 0) {
        return {
          role: 'assistant',
          content: msg.content,
          name: msg.name,
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
          parts.push({
            type: 'text',
            text: contentPart.text,
          } as TextPart);
        } else if (contentPart.type === 'refusal') {
          parts.push({
            type: 'refusal',
            reason: contentPart.refusal,
          } as RefusalPart);
        }
      });
    }

    return {
      type: 'session_message',
      role: 'assistant',
      content: parts,
      name: msg.name,
      refusal: msg.refusal,
    } as SessionMessage;
  }

  private getFromUserMessageParam(msg: ChatCompletionUserMessageParam): SessionMessage {
    if (typeof msg.content === 'string') {
      return {
        type: 'session_message',
        role: 'user',
        content: msg.content,
        name: msg.name,
      } as SessionMessage;
    }

    const parts = [] as ContentPart[];
    msg.content.forEach((contentPart) => {
      if (contentPart.type === 'text') {
        parts.push({
          type: 'text',
          text: contentPart.text,
        } as TextPart);
      } else if (contentPart.type === 'image_url') {
        const isBase64 = contentPart.image_url.url.startsWith('data:image/');
        if (isBase64) {
          const mediaType = contentPart.image_url.url.split(';')[0].split(':')[1];
          const base64Data = contentPart.image_url.url.split(',')[1];
        parts.push({
          type: 'image',
          image: {
            sourceType: 'base64',
            data: base64Data,
            mimeType: mediaType,
          },
        } as ImagePart);}
        else {
        parts.push({
          type: 'image',
          image: {
            sourceType: 'url',
            url: contentPart.image_url.url,
          },
        } as ImagePart);}
      } else if (contentPart.type === 'input_audio') {
        parts.push({
          type: 'audio',
          audio: {
            sourceType: 'base64',
            data: contentPart.input_audio.data,
            mimeType: `audio/${contentPart.input_audio.format}`,
          },
        } as AudioPart);
      } else if (contentPart.type === 'file') {
        parts.push({
          type: 'document',
          document: {
            sourceType: contentPart.file.file_data
              ? 'base64'
              : contentPart.file.file_id
                ? 'file_id'
                : 'url', // fallback to url, though it's not supported in current implementation
            data: contentPart.file.file_data ?? undefined,
            fileId: contentPart.file.file_id ?? undefined,
            fileName: contentPart.file.filename ?? undefined,
            mediaType: contentPart.file.file_data ? '' : undefined,
          },
        } as DocumentPart);
      }
    });

    return {
      type: 'session_message',
      role: 'user',
      content: parts,
      name: msg.name,
    } as SessionMessage;
  }

  private getFromToolMessageParam(msg: ChatCompletionToolMessageParam): SessionMessage {
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
        } else {
          toolResultParts.push({
            type: 'tool_result',
            id: msg.tool_call_id,
            result: JSON.parse(contentPart.text),
          } as ToolResultPart);
        }
      });
    }

    return {
      type: 'session_message',
      role: 'tool',
      content: toolResultParts,
    } as SessionMessage;
  }

  private getFromFunctionMessageParam(msg: ChatCompletionFunctionMessageParam): SessionMessage {
    return {
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
    } as SessionMessage;
  }

  convertFromRawMessage(rawMsg: ChatCompletionMessageParam): SessionMessage {
    const message = (() => {
      switch (rawMsg.role) {
        case 'assistant':
          return this.getFromAssistantMessageParam(rawMsg as ChatCompletionAssistantMessageParam);
        case 'user':
          return this.getFromUserMessageParam(rawMsg as ChatCompletionUserMessageParam);
        case 'tool':
          return this.getFromToolMessageParam(rawMsg as ChatCompletionToolMessageParam);
        case 'function':
          return this.getFromFunctionMessageParam(rawMsg as ChatCompletionFunctionMessageParam);
        default:
          throw new InvalidMessageError(`Unsupported message role: ${rawMsg.role}`);
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
            .filter((part) => part.type === 'text')
            .map((part) => part as TextPart)
            .map(
              (part) =>
                ({
                  type: 'text',
                  text: part.text,
                }) as ChatCompletionContentPartText | ChatCompletionContentPartRefusal,
            )
            .concat(
              msg.content
                .filter((part) => part.type === 'refusal')
                .map((part) => part as RefusalPart)
                .map(
                  (part) =>
                    ({
                      type: 'refusal',
                      refusal: part.reason,
                    }) as ChatCompletionContentPartText | ChatCompletionContentPartRefusal,
                ),
            )
        : (msg.content ?? null);

    return {
      role: 'assistant',
      content: content instanceof Array && content.length === 0 ? null : content,
      tool_calls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
      audio: null, // todo: may need to handle this in the future.
      name: msg.name,
      refusal: msg.refusal,
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

    const textParts = msg.content
      .filter((part) => part.type === 'text')
      .map((part) => part as TextPart)
      .map(
        (part) =>
          ({
            type: 'text',
            text: part.text,
          }) as ChatCompletionContentPartText,
      );
    const imageParts = msg.content
      .filter((part) => part.type === 'image')
      .map((part) => part as ImagePart)
      .map(
        (part) => {
          switch (part.image.sourceType) {
            case 'url':
              return {
                type: 'image_url',
                image_url: { url: part.image.url },
              } as ChatCompletionContentPartImage;
            case 'base64':
              return {
                type: 'image_url',
                image_url: { url: `data:${part.image.mimeType};base64,${part.image.data}` },
              } as ChatCompletionContentPartImage;
            default:
              throw new InvalidMessageError(
                `Unsupported image sourceType: ${part.image.sourceType}`,
              );
          }
        }
      );
    const audioParts = msg.content
      .filter((part) => part.type === 'audio')
      .map((part) => part as AudioPart)
      .filter(
        (part) =>
          part.audio.sourceType === 'base64' &&
          ['audio/wav', 'audio/mp3'].includes(part.audio.mimeType),
      )
      .map(
        (part) =>
          ({
            type: 'input_audio',
            input_audio: {
              data: part.audio.sourceType === 'base64' ? part.audio.data : '',
              format:
                part.audio.sourceType === 'base64' ? part.audio.mimeType.replace('audio/', '') : '',
            },
          }) as ChatCompletionContentPartInputAudio,
      );
    const fileParts = msg.content
      .filter((part) => part.type === 'document')
      .map((part) => part as DocumentPart)
      .map(
        (part) =>
          ({
            type: 'file',
            file: {
              file_data: part.document.sourceType === 'base64' ? part.document.data : undefined,
              file_id: part.document.sourceType === 'file_id' ? part.document.fileId : undefined,
              file_name: part.document.sourceType === 'base64' ? part.document.fileName : undefined,
            },
          }) as ChatCompletionContentPart.File,
      );
    const content = [...textParts, ...imageParts, ...audioParts, ...fileParts];

    return {
      role: 'user',
      content: content.length === 0 ? null : content,
      name: msg.name,
    } as ChatCompletionUserMessageParam;
  }

  private getToolMessageParam(msg: SessionMessage): ChatCompletionToolMessageParam {
    if (msg.role !== 'tool') {
      throw new InvalidMessageError(`Message role must be tool, but got ${msg.role}`);
    }

    if (!msg.content || typeof msg.content === 'string' || msg.content.length === 0) {
      throw new InvalidMessageError('Tool message content must be a non-empty array of content parts.');
    }

    const toolResults = msg.content
      .filter((part) => part.type === 'tool_result')
      .map((part) => part as ToolResultPart);

    if (toolResults.length !== 1) {
      throw new InvalidMessageError('Tool message content must have exactly one tool result part.');
    }

    const toolResult = toolResults[0];
    return {
      role: 'tool',
      tool_call_id: toolResult.id,
      content: toolResult.result,
    };
  }

  convertToRawMessage(msg: SessionMessage): ChatCompletionMessageParam {
    const rawMsg = (() => {
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
    return this.dialectResolver ? this.dialectResolver.manipulateRawMessage(rawMsg, msg) : rawMsg;
  }

  async generate(
    options: ModelOptions,
    messages: Array<ChatCompletionMessageParam>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<ChatCompletionMessageParam>> {
    const params = this.getChatCompletionStreamParams(options, messages);

    let textChunkStarted = false;
    let thinkingChunkStarted = false;
    let thinkingText = '';

    try {
      const response = await this.client.chat.completions
        .stream(params, requestOptions)
        .on('chunk', (chunk) => {
          const choice = chunk.choices[0];
          if (!choice) return;

          const delta = choice.delta;

          // Handle thinking/reasoning content (provider-specific field, extracted via dialect)
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

          // Handle text content
          const textDelta: string | null | undefined = delta.content;
          if (textDelta) {
            // If thinking was active, signal its end before text begins
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

          // Handle stream finish
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
          if (callbacks?.onText) {
            callbacks.onText(content);
          }
        })
        .on('finalFunctionToolCall', (toolCall) => {
          if (callbacks?.onToolUse) {
            callbacks.onToolUse(toolCall.name, toolCall.arguments);
          }
        })
        .on('error', (error) => {
          if (callbacks?.onError) {
            callbacks.onError(error);
          }
        })
        .on('abort', (error) => {
          if (callbacks?.onError) {
            callbacks.onError(error);
          }
        })
        .finalChatCompletion();

      return {
        responseId: response.id,
        responseMessage: response.choices[0].message as ChatCompletionMessageParam,
        responseModel: response.model,
        stopReason:
          this.dialectResolver?.extractFromResponse('stop_reason', response) ??
          this.getStopReason(response.choices[0].finish_reason),
        refusal: response.choices[0].message.refusal ?? undefined,
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
      } as ModelResponse<ChatCompletionMessageParam>;
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
    return new ModelRequestTimeoutError('openai', requestOptions?.timeout, error);
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
    messages: Array<ChatCompletionMessageParam>,
  ): ChatCompletionStreamParams {
    const inputs: ChatCompletionMessageParam[] = [];
    if (options.systemPrompt) {
      inputs.push({
        role: 'developer',
        content: options.systemPrompt,
      });
    }
    inputs.push(...messages);

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
