import { OpenAI } from 'openai/client';
import type {
  Response,
  ResponseCreateParamsNonStreaming,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputMessage,
  ResponseStreamEvent,
  ToolChoiceOptions,
} from 'openai/resources/responses/responses';
import {
  ApiModes,
  ContentPart,
  DocumentPart,
  ImagePart,
  RawDeltaMessageType,
  RawLLMParametersType,
  RawResponseType,
  RefusalPart,
  SessionMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from '../message';
import {
  AgentInterruptedError,
  InvalidMessageError,
  ModelRequestTimeoutError,
} from '../errors/types';
import { isLikelyAbortError, isLikelyTimeoutError } from '../errors/guards';
import {
  DialectResolver,
  ModelOptions,
  ModelRequestOptions,
  ModelResponse,
  ProviderTransport,
  StreamCallbacks,
} from './base';

export class ResponseTransport extends ProviderTransport<ResponseInputItem> {
  readonly client: OpenAI;
  private static readonly OFFICIAL_BASE_URL = 'https://api.openai.com/v1';
  private readonly configuredBaseUrl?: string;

  private static normalizeBaseUrl(baseUrl?: string): string | undefined {
    if (!baseUrl) return baseUrl;
    const normalized = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    // Some docs provide the full responses endpoint; SDK baseURL should be the parent API root.
    if (normalized.toLowerCase().endsWith('/responses')) {
      return normalized.slice(0, -'/responses'.length);
    }
    return normalized;
  }

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<
      ResponseInputItem,
      RawDeltaMessageType,
      RawLLMParametersType,
      RawResponseType
    >,
  ) {
    super(ApiModes.RESPONSE, dialectResolver);
    this.configuredBaseUrl = ResponseTransport.normalizeBaseUrl(baseUrl);
    this.client = new OpenAI({
      baseURL: this.configuredBaseUrl,
      apiKey,
    });
  }

  private createSyntheticId(prefix: string): string {
    return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  }

  private convertResponseInputContentItem(
    content: ResponseInputItem.Message['content'][number] | ResponseOutputMessage['content'][number],
  ): ContentPart | undefined {
    if (content.type === 'input_text' || content.type === 'output_text') {
      return {
        type: 'text',
        text: content.text,
      } as TextPart;
    }

    if (content.type === 'refusal') {
      return {
        type: 'refusal',
        reason: content.refusal,
      } as RefusalPart;
    }

    if (content.type === 'input_image') {
      if (content.image_url) {
        return {
          type: 'image',
          image: {
            sourceType: 'url',
            url: content.image_url,
          },
        } as ImagePart;
      }
      if (content.file_id) {
        return {
          type: 'image',
          image: {
            sourceType: 'file_id',
            fileId: content.file_id,
          },
        } as ImagePart;
      }
      return undefined;
    }

    if (content.type === 'input_file') {
      if (content.file_url) {
        return {
          type: 'document',
          document: {
            sourceType: 'url',
            url: content.file_url,
          },
        } as DocumentPart;
      }
      if (content.file_data) {
        return {
          type: 'document',
          document: {
            sourceType: 'base64',
            data: content.file_data,
            fileName: content.filename ?? '',
          },
        } as DocumentPart;
      }
      if (content.file_id) {
        return {
          type: 'document',
          document: {
            sourceType: 'file_id',
            fileId: content.file_id,
          },
        } as DocumentPart;
      }
      return undefined;
    }

    return undefined;
  }

  private getFromMessage(rawMsg: ResponseInputItem): SessionMessage {
    if (rawMsg.type !== 'message') {
      throw new InvalidMessageError(`Expected message item, received: ${rawMsg.type}`);
    }

    const parts: ContentPart[] = [];
    if (typeof rawMsg.content === 'string') {
      parts.push({
        type: 'text',
        text: rawMsg.content,
      } as TextPart);
    } else {
      for (const content of rawMsg.content) {
        const part = this.convertResponseInputContentItem(content);
        if (part) {
          parts.push(part);
        }
      }
    }

    const role = rawMsg.role === 'assistant' ? 'assistant' : 'user';
    return {
      type: 'session_message',
      role,
      content: parts,
    } as SessionMessage;
  }

  private getFromFunctionCall(rawMsg: ResponseFunctionToolCall): SessionMessage {
    return {
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: rawMsg.call_id,
          name: rawMsg.name,
          arguments: rawMsg.arguments,
        } as ToolCallPart,
      ],
    } as SessionMessage;
  }

  private getFromFunctionCallOutput(rawMsg: ResponseInputItem.FunctionCallOutput): SessionMessage {
    const result =
      typeof rawMsg.output === 'string'
        ? rawMsg.output
        : JSON.stringify(
            rawMsg.output.map((item) => {
              if (item.type === 'input_text') {
                return item.text;
              }
              if (item.type === 'input_image') {
                return item.image_url ?? item.file_id ?? '';
              }
              return item.file_id ?? item.file_url ?? item.filename ?? '';
            }),
          );

    return {
      type: 'session_message',
      role: 'tool',
      content: [
        {
          type: 'tool_result',
          id: rawMsg.call_id,
          result,
        } as ToolResultPart,
      ],
    } as SessionMessage;
  }

  convertFromRawMessage(rawMsg: ResponseInputItem): SessionMessage {
    const message = (() => {
      if (rawMsg.type === 'function_call') {
        return this.getFromFunctionCall(rawMsg);
      }
      if (rawMsg.type === 'function_call_output') {
        return this.getFromFunctionCallOutput(rawMsg);
      }
      if (rawMsg.type === 'message' || (!('type' in rawMsg) && 'role' in rawMsg)) {
        if (rawMsg.role === 'system' || rawMsg.role === 'developer') {
          throw new InvalidMessageError(
            'System/developer role messages must be provided via instructions, not message items.',
          );
        }
        return this.getFromMessage(rawMsg as ResponseInputItem.Message);
      }
      throw new InvalidMessageError(`Unsupported response input item type: ${rawMsg.type}`);
    })();

    return this.dialectResolver ? this.dialectResolver.manipulateMessage(message, rawMsg) : message;
  }

  private getUserMessage(msg: SessionMessage): ResponseInputItem.Message {
    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw new InvalidMessageError('User message must have content');
    }

    if (typeof msg.content === 'string') {
      return {
        type: 'message',
        role: 'user',
        content: [{ type: 'input_text', text: msg.content }],
      };
    }

    const content = [] as ResponseInputItem.Message['content'];
    for (const part of msg.content) {
      if (part.type === 'text') {
        content.push({ type: 'input_text', text: (part as TextPart).text });
      } else if (part.type === 'image') {
        const image = (part as ImagePart).image;
        if (image.sourceType === 'url') {
          content.push({ type: 'input_image', detail: 'auto', image_url: image.url });
        } else if (image.sourceType === 'base64') {
          content.push({
            type: 'input_image',
            detail: 'auto',
            image_url: `data:${image.mimeType};base64,${image.data}`,
          });
        } else {
          content.push({ type: 'input_image', detail: 'auto', file_id: image.fileId });
        }
      } else if (part.type === 'document') {
        const doc = (part as DocumentPart).document;
        if (doc.sourceType === 'url') {
          content.push({ type: 'input_file', file_url: doc.url });
        } else if (doc.sourceType === 'base64') {
          content.push({
            type: 'input_file',
            file_data: doc.data,
            filename: doc.fileName,
          });
        } else {
          content.push({ type: 'input_file', file_id: doc.fileId });
        }
      }
    }

    if (content.length === 0) {
      throw new InvalidMessageError('User message must include at least one supported content part');
    }

    return {
      type: 'message',
      role: 'user',
      content,
    };
  }

  private getAssistantMessage(msg: SessionMessage): ResponseInputItem {
    if (msg.content instanceof Array) {
      const toolCalls = msg.content
        .filter((part) => part.type === 'tool_call')
        .map((part) => part as ToolCallPart);

      if (toolCalls.length > 0) {
        const toolCall = toolCalls[0];
        return {
          type: 'function_call',
          id: this.createSyntheticId('fc'),
          call_id: toolCall.id,
          name: toolCall.name,
          arguments: toolCall.arguments,
          status: 'completed',
        };
      }

      const messageContent = msg.content
        .filter((part) => part.type === 'text' || part.type === 'refusal')
        .map((part) => {
          if (part.type === 'text') {
            return {
              type: 'output_text',
              text: (part as TextPart).text,
              annotations: [],
            };
          }
          return {
            type: 'refusal',
            refusal: (part as RefusalPart).reason,
          };
        });

      return {
        type: 'message',
        id: this.createSyntheticId('msg'),
        role: 'assistant',
        status: 'completed',
        content: messageContent,
      } as ResponseOutputMessage;
    }

    return {
      type: 'message',
      id: this.createSyntheticId('msg'),
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text: msg.content ?? '', annotations: [] }],
    } as ResponseOutputMessage;
  }

  private getToolMessage(msg: SessionMessage): ResponseInputItem | ResponseInputItem[] {
    if (!msg.content || typeof msg.content === 'string' || msg.content.length === 0) {
      throw new InvalidMessageError('Tool message must have non-empty tool_result content');
    }

    const toolResults = msg.content
      .filter((part) => part.type === 'tool_result')
      .map((part) => part as ToolResultPart);
    if (toolResults.length === 0) {
      throw new InvalidMessageError('Tool message must include at least one tool_result part');
    }

    const rawMessages: ResponseInputItem[] = toolResults.map(
      (toolResult) =>
        ({
          type: 'function_call_output',
          call_id: toolResult.id,
          output: toolResult.result,
          status: 'completed',
        }) as ResponseInputItem.FunctionCallOutput,
    );

    const otherParts = msg.content.filter((part) => part.type !== 'tool_result');
    if (otherParts.length > 0) {
      rawMessages.push(
        this.getUserMessage({
          ...msg,
          role: 'user',
          content: otherParts,
        } as SessionMessage),
      );
    }

    return rawMessages.length === 1 ? rawMessages[0] : rawMessages;
  }

  convertToRawMessage(msg: SessionMessage): ResponseInputItem | ResponseInputItem[] {
    const raw = (() => {
      switch (msg.role) {
        case 'assistant':
          return this.getAssistantMessage(msg);
        case 'user':
          return this.getUserMessage(msg);
        case 'tool':
          return this.getToolMessage(msg);
        default:
          throw new InvalidMessageError(`Unsupported message role: ${msg.role}`);
      }
    })();

    if (!this.dialectResolver) {
      return raw;
    }
    if (Array.isArray(raw)) {
      const manipulated = raw.map((rawMsg) => this.dialectResolver!.manipulateRawMessage(rawMsg, msg));
      return manipulated.length === 1 ? manipulated[0] : manipulated;
    }
    return this.dialectResolver.manipulateRawMessage(raw, msg);
  }

  private getToolChoice(options: ModelOptions): ToolChoiceOptions | 'required' | 'none' | undefined {
    if (!options.tools || options.tools.length === 0) {
      return undefined;
    }

    switch (options.toolChoice) {
      case 'force':
        return 'required';
      case 'none':
        return 'none';
      default:
        return 'auto';
    }
  }

  private getReasoningEffort(thinkEffort?: ModelOptions['thinkEffort']):
    | 'none'
    | 'low'
    | 'medium'
    | 'high'
    | 'xhigh'
    | undefined {
    switch (thinkEffort) {
      case 'disable':
        return 'none';
      case 'low':
        return 'low';
      case 'medium':
        return 'medium';
      case 'high':
        return 'high';
      case 'extreme':
        return 'xhigh';
      default:
        return undefined;
    }
  }

  private getStopReason(response: Response): string {
    const hasToolCall = response.output.some((item) => item.type === 'function_call');
    if (hasToolCall) {
      return 'tool_call';
    }

    if (response.status === 'cancelled') {
      return 'cancelled';
    }

    if (response.status === 'failed') {
      return 'failed';
    }

    if (response.status === 'queued' || response.status === 'in_progress') {
      return response.status;
    }

    if (response.status === 'incomplete') {
      switch (response.incomplete_details?.reason) {
        case 'max_output_tokens':
          return 'max_tokens';
        case 'content_filter':
          return 'refusal';
        default:
          return 'max_tokens';
      }
    }

    return 'stop';
  }

  private getResponseMessage(response: Response): ResponseInputItem {
    const functionCall = response.output.find((item) => item.type === 'function_call');
    if (functionCall && functionCall.type === 'function_call') {
      return functionCall;
    }

    const outputMessage = response.output.find((item) => item.type === 'message');
    if (outputMessage && outputMessage.type === 'message') {
      return outputMessage;
    }

    return {
      type: 'message',
      id: this.createSyntheticId('msg'),
      role: 'assistant',
      status: 'completed',
      content: [{ type: 'output_text', text: response.output_text ?? '', annotations: [] }],
    } as ResponseOutputMessage;
  }

  private getRefusal(responseMessage: ResponseInputItem): string | undefined {
    if (responseMessage.type !== 'message') {
      return undefined;
    }
    if (!Array.isArray(responseMessage.content)) {
      return undefined;
    }

    const refusal = responseMessage.content.find((item) => item.type === 'refusal');
    return refusal?.type === 'refusal' ? refusal.refusal : undefined;
  }

  private buildParams(
    options: ModelOptions,
    messages: Array<ResponseInputItem>,
  ): ResponseCreateParamsNonStreaming {
    const params: ResponseCreateParamsNonStreaming = {
      model: options.model,
      input: messages,
      instructions: options.systemPrompt,
      max_output_tokens: options.maxTokens,
      temperature: options.temperature,
      metadata: options.metadata,
      parallel_tool_calls: options.parallelToolCalls,
      tool_choice: this.getToolChoice(options),
      store: false,
      reasoning: {
        effort: this.getReasoningEffort(options.thinkEffort),
      },
      text: options.outputSchema
        ? {
            format: {
              type: 'json_schema',
              name: 'structured_output',
              schema: options.outputSchema.schema,
              strict: options.outputSchema.strict,
            },
          }
        : undefined,
      tools: options.tools?.map((tool) => ({
        type: 'function' as const,
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
        strict: true,
      })),
    };

    return params;
  }

  private isOfficialService(): boolean {
    const normalized = ResponseTransport.normalizeBaseUrl(this.configuredBaseUrl)?.toLowerCase();
    return !normalized || normalized === ResponseTransport.OFFICIAL_BASE_URL;
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

  private async generateWithOfficialStream(
    params: ResponseCreateParamsNonStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<ResponseInputItem>> {
    let textChunkStarted = false;
    let textBuffer = '';
    let sawTextDelta = false;
    let thinkingChunkStarted = false;
    let thinkingBuffer = '';
    let sawThinkingDelta = false;

    const closeThinkingChunk = (): void => {
      if (!thinkingChunkStarted) {
        return;
      }
      thinkingChunkStarted = false;
      callbacks?.onThinkingChunk?.('', 'end');
      if (thinkingBuffer) {
        callbacks?.onThinking?.(thinkingBuffer);
      }
    };

    let response: Response | undefined;
    const stream = this.client.responses
      .stream({ ...params, stream: true }, this.getStreamRequestOptions(requestOptions))
      .on('response.reasoning_text.delta', (event) => {
        sawThinkingDelta = true;
        thinkingBuffer += event.delta;
        if (!thinkingChunkStarted) {
          thinkingChunkStarted = true;
          callbacks?.onThinkingChunk?.(event.delta, 'begin');
        } else {
          callbacks?.onThinkingChunk?.(event.delta);
        }
      })
      .on('response.reasoning_summary_text.delta', (event) => {
        sawThinkingDelta = true;
        thinkingBuffer += event.delta;
        if (!thinkingChunkStarted) {
          thinkingChunkStarted = true;
          callbacks?.onThinkingChunk?.(event.delta, 'begin');
        } else {
          callbacks?.onThinkingChunk?.(event.delta);
        }
      })
      .on('response.reasoning_text.done', (event) => {
        if (!sawThinkingDelta && event.text) {
          thinkingBuffer += event.text;
          if (!thinkingChunkStarted) {
            thinkingChunkStarted = true;
            callbacks?.onThinkingChunk?.(event.text, 'begin');
          } else {
            callbacks?.onThinkingChunk?.(event.text);
          }
        }
      })
      .on('response.reasoning_summary_text.done', (event) => {
        if (!sawThinkingDelta && event.text) {
          thinkingBuffer += event.text;
          if (!thinkingChunkStarted) {
            thinkingChunkStarted = true;
            callbacks?.onThinkingChunk?.(event.text, 'begin');
          } else {
            callbacks?.onThinkingChunk?.(event.text);
          }
        }
      })
      .on('response.output_text.delta', (event) => {
        closeThinkingChunk();
        sawTextDelta = true;
        textBuffer += event.delta;
        if (!textChunkStarted) {
          textChunkStarted = true;
          callbacks?.onTextChunk?.(event.delta, 'begin');
        } else {
          callbacks?.onTextChunk?.(event.delta);
        }
      })
      .on('response.output_text.done', (event) => {
        closeThinkingChunk();
        if (!sawTextDelta && !textChunkStarted) {
          textChunkStarted = true;
          callbacks?.onTextChunk?.(event.text, 'begin');
        } else if (!sawTextDelta) {
          callbacks?.onTextChunk?.(event.text);
        }
        if (!sawTextDelta) {
          textBuffer += event.text;
        }
      })
      .on('response.function_call_arguments.done', (event) => {
        callbacks?.onToolUse?.(event.name, event.arguments);
      })
      .on('response.completed', (event) => {
        response = event.response;
      })
      .on('response.incomplete', (event) => {
        response = event.response;
      })
      .on('response.failed', (event) => {
        response = event.response;
      })
      .on('error', (error) => {
        callbacks?.onError?.(error instanceof Error ? error : new Error(String(error)));
      });

    const finalResponse = await stream.finalResponse();
    response = response ?? (finalResponse as Response);

    closeThinkingChunk();

    if (textChunkStarted) {
      callbacks?.onTextChunk?.('', 'end');
    } else if (response.output_text) {
      callbacks?.onTextChunk?.(response.output_text, 'begin');
      callbacks?.onTextChunk?.('', 'end');
      textBuffer = response.output_text;
    }

    if (textBuffer) {
      callbacks?.onText?.(textBuffer);
    }

    const responseMessage = this.getResponseMessage(response);
    return {
      responseId: response.id,
      responseMessage,
      responseModel: response.model,
      stopReason: this.getStopReason(response),
      refusal: this.getRefusal(responseMessage),
      usage: response.usage
        ? {
            inputTokens: response.usage.input_tokens,
            outputTokens: response.usage.output_tokens,
            totalTokens: response.usage.total_tokens,
            reasoningTokens: response.usage.output_tokens_details?.reasoning_tokens,
            cacheReadTokens: response.usage.input_tokens_details?.cached_tokens,
            inputTokenDetails: response.usage.input_tokens_details
              ? { ...response.usage.input_tokens_details }
              : undefined,
            outputTokenDetails: response.usage.output_tokens_details
              ? { ...response.usage.output_tokens_details }
              : undefined,
          }
        : undefined,
    };
  }

  private async generateWithThirdPartyCreate(
    params: ResponseCreateParamsNonStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<ResponseInputItem>> {
    let textChunkStarted = false;
    let textBuffer = '';
    let sawTextDelta = false;
    let thinkingChunkStarted = false;
    let thinkingBuffer = '';
    let sawThinkingDelta = false;

    const closeThinkingChunk = (): void => {
      if (!thinkingChunkStarted) {
        return;
      }
      thinkingChunkStarted = false;
      callbacks?.onThinkingChunk?.('', 'end');
      if (thinkingBuffer) {
        callbacks?.onThinking?.(thinkingBuffer);
      }
    };

    const stream = await this.client.responses.create(
      { ...params, stream: true },
      this.getStreamRequestOptions(requestOptions),
    );

    let response: Response | undefined;
    for await (const event of stream) {
      const type = (event as { type?: string }).type;

      if (type === 'response.keep_alive') {
        continue;
      }

      switch (type) {
        case 'response.reasoning_text.delta': {
          const deltaEvent = event as Extract<ResponseStreamEvent, { type: 'response.reasoning_text.delta' }>;
          sawThinkingDelta = true;
          thinkingBuffer += deltaEvent.delta;
          if (!thinkingChunkStarted) {
            thinkingChunkStarted = true;
            callbacks?.onThinkingChunk?.(deltaEvent.delta, 'begin');
          } else {
            callbacks?.onThinkingChunk?.(deltaEvent.delta);
          }
          break;
        }
        case 'response.reasoning_summary_text.delta': {
          const deltaEvent = event as Extract<ResponseStreamEvent, { type: 'response.reasoning_summary_text.delta' }>;
          sawThinkingDelta = true;
          thinkingBuffer += deltaEvent.delta;
          if (!thinkingChunkStarted) {
            thinkingChunkStarted = true;
            callbacks?.onThinkingChunk?.(deltaEvent.delta, 'begin');
          } else {
            callbacks?.onThinkingChunk?.(deltaEvent.delta);
          }
          break;
        }
        case 'response.reasoning_text.done': {
          const doneEvent = event as Extract<ResponseStreamEvent, { type: 'response.reasoning_text.done' }>;
          if (!sawThinkingDelta && doneEvent.text) {
            thinkingBuffer += doneEvent.text;
            if (!thinkingChunkStarted) {
              thinkingChunkStarted = true;
              callbacks?.onThinkingChunk?.(doneEvent.text, 'begin');
            } else {
              callbacks?.onThinkingChunk?.(doneEvent.text);
            }
          }
          break;
        }
        case 'response.reasoning_summary_text.done': {
          const doneEvent = event as Extract<ResponseStreamEvent, { type: 'response.reasoning_summary_text.done' }>;
          if (!sawThinkingDelta && doneEvent.text) {
            thinkingBuffer += doneEvent.text;
            if (!thinkingChunkStarted) {
              thinkingChunkStarted = true;
              callbacks?.onThinkingChunk?.(doneEvent.text, 'begin');
            } else {
              callbacks?.onThinkingChunk?.(doneEvent.text);
            }
          }
          break;
        }
        case 'response.output_text.delta': {
          const deltaEvent = event as Extract<ResponseStreamEvent, { type: 'response.output_text.delta' }>;
          closeThinkingChunk();
          sawTextDelta = true;
          textBuffer += deltaEvent.delta;
          if (!textChunkStarted) {
            textChunkStarted = true;
            callbacks?.onTextChunk?.(deltaEvent.delta, 'begin');
          } else {
            callbacks?.onTextChunk?.(deltaEvent.delta);
          }
          break;
        }
        case 'response.output_text.done': {
          const doneEvent = event as Extract<ResponseStreamEvent, { type: 'response.output_text.done' }>;
          closeThinkingChunk();
          if (!sawTextDelta && !textChunkStarted) {
            textChunkStarted = true;
            callbacks?.onTextChunk?.(doneEvent.text, 'begin');
          } else if (!sawTextDelta) {
            callbacks?.onTextChunk?.(doneEvent.text);
          }
          if (!sawTextDelta) {
            textBuffer += doneEvent.text;
          }
          break;
        }
        case 'response.function_call_arguments.done': {
          const toolEvent = event as Extract<ResponseStreamEvent, { type: 'response.function_call_arguments.done' }>;
          callbacks?.onToolUse?.(toolEvent.name, toolEvent.arguments);
          break;
        }
        case 'error': {
          const errEvent = event as Extract<ResponseStreamEvent, { type: 'error' }>;
          callbacks?.onError?.(new Error(errEvent.message ?? 'Unknown response stream error'));
          break;
        }
        case 'response.completed':
        case 'response.incomplete':
        case 'response.failed': {
          response = (event as { response: Response }).response;
          break;
        }
        default:
          break;
      }
    }

    if (!response) {
      throw new Error('Response stream ended without a terminal response event');
    }

    closeThinkingChunk();

    if (textChunkStarted) {
      callbacks?.onTextChunk?.('', 'end');
    } else if (response.output_text) {
      callbacks?.onTextChunk?.(response.output_text, 'begin');
      callbacks?.onTextChunk?.('', 'end');
      textBuffer = response.output_text;
    }

    if (textBuffer) {
      callbacks?.onText?.(textBuffer);
    }

    const responseMessage = this.getResponseMessage(response);
    return {
      responseId: response.id,
      responseMessage,
      responseModel: response.model,
      stopReason: this.getStopReason(response),
      refusal: this.getRefusal(responseMessage),
      usage: response.usage
        ? {
            inputTokens: response.usage.input_tokens,
            outputTokens: response.usage.output_tokens,
            totalTokens: response.usage.total_tokens,
            reasoningTokens: response.usage.output_tokens_details?.reasoning_tokens,
            cacheReadTokens: response.usage.input_tokens_details?.cached_tokens,
            inputTokenDetails: response.usage.input_tokens_details
              ? { ...response.usage.input_tokens_details }
              : undefined,
            outputTokenDetails: response.usage.output_tokens_details
              ? { ...response.usage.output_tokens_details }
              : undefined,
          }
        : undefined,
    };
  }

  async generate(
    options: ModelOptions,
    messages: Array<ResponseInputItem>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<ResponseInputItem>> {
    const params = this.buildParams(options, messages);

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
}