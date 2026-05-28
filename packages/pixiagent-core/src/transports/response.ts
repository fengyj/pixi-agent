import { OpenAI } from 'openai/client';
import type {
  Response,
  ResponseCreateParamsStreaming,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputMessage,
  ResponseStreamEvent,
  ToolChoiceOptions,
} from 'openai/resources/responses/responses';
import {
  ApiModes,
  ContentPart,
  ImagePart,
  RawDeltaMessageType,
  RawLLMParametersType,
  RawResponseType,
  RefusalPart,
  SessionMessage,
  TextPart,
  ToolCallPart,
  ToolResultPart,
  ResponseApiMessage,
  ThinkingPart,
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

export class ResponseTransport extends ProviderTransport<ResponseApiMessage> {
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
      ResponseApiMessage,
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

  private convertResponseInputContentItem(
    content:
      | ResponseInputItem.Message['content'][number]
      | ResponseOutputMessage['content'][number],
  ): ContentPart | undefined {
    if (content.type === 'input_text' || content.type === 'output_text') {
      return this.convertFromResponseTextContent(content.text);
    }

    if (content.type === 'refusal') {
      return this.convertFromResponseRefusalContent(content.refusal);
    }

    if (content.type === 'input_image') {
      return this.convertFromResponseImageContent(content);
    }

    return undefined;
  }

  private convertResponseInputItemToParts(item: ResponseInputItem): ContentPart[] {
    if (item.type === 'message') {
      const contents = typeof item.content === 'string' ? [item.content] : item.content;
      return contents
        .map((content) =>
          typeof content === 'string'
            ? this.convertFromResponseTextContent(content)
            : this.convertResponseInputContentItem(content),
        )
        .filter((part): part is ContentPart => part !== undefined);
    }

    if (item.type === 'function_call') {
      return [
        {
          type: 'tool_call',
          id: item.call_id,
          name: item.name,
          arguments: item.arguments,
        } as ToolCallPart,
      ];
    }

    if (
      item.type === 'computer_call' ||
      item.type === 'file_search_call' ||
      item.type === 'custom_tool_call' ||
      item.type === 'tool_search_call' ||
      item.type === 'web_search_call' ||
      item.type === 'code_interpreter_call' ||
      item.type === 'image_generation_call' ||
      item.type === 'local_shell_call' ||
      item.type === 'shell_call' ||
      item.type === 'apply_patch_call' ||
      item.type === 'mcp_call' ||
      item.type === 'mcp_list_tools' ||
      item.type === 'mcp_approval_request' ||
      item.type === 'mcp_approval_response'
    ) {
      const itemAny = item as { type: string; id?: string; custom?: { name?: string } };
      const toolName =
        item.type === 'custom_tool_call'
          ? itemAny.custom?.name ?? item.type
          : item.type;
      const toolId =
        itemAny.id && typeof itemAny.id === 'string'
          ? itemAny.id
          : `${item.type}:${Math.random().toString(36).slice(2, 10)}`;
      return [
        {
          type: 'tool_call',
          id: toolId,
          name: toolName,
          arguments: ContentPart.createProviderToolCallArguments('openai_response', item.type, item),
        } as ToolCallPart,
      ];
    }

    if (item.type === 'function_call_output') {
      return [
        {
          type: 'tool_result',
          id: item.call_id,
          result: this.convertFunctionCallOutput(item.output),
        } as ToolResultPart,
      ];
    }

    if (item.type === 'reasoning') {
      const content = item.summary?.map((s) => s.text).join('') ?? '';
      if (!content) return [];
      return [
        {
          type: 'thinking',
          content,
        } as ThinkingPart,
      ];
    }

    return [];
  }

  private convertFunctionCallOutput(
    output: ResponseInputItem.FunctionCallOutput['output'],
  ): string | Array<TextPart | ImagePart> {
    if (typeof output === 'string') {
      return output;
    }
    return output
      .map((item) => {
        if (item.type === 'input_text') {
          return { type: 'text', text: item.text } as TextPart;
        }
        if (item.type === 'input_image') {
          if (item.image_url) {
            return {
              type: 'image',
              image: { sourceType: 'url', url: item.image_url },
            } as ImagePart;
          }
          if (item.file_id) {
            return {
              type: 'image',
              image: { sourceType: 'file_id', fileId: item.file_id },
            } as ImagePart;
          }
        }
        return null;
      })
      .filter((part): part is TextPart | ImagePart => part !== null);
  }

  private convertFromResponseTextContent(text: string): TextPart {
    return {
      type: 'text',
      text,
    };
  }

  private convertFromResponseRefusalContent(reason: string): RefusalPart {
    return {
      type: 'refusal',
      reason,
    };
  }

  private convertFromResponseImageContent(
    content: Extract<ResponseInputItem.Message['content'][number], { type: 'input_image' }>,
  ): ImagePart | undefined {
    if (content.image_url) {
      return {
        type: 'image',
        image: {
          sourceType: 'url',
          url: content.image_url,
        },
      };
    }
    if (content.file_id) {
      return {
        type: 'image',
        image: {
          sourceType: 'file_id',
          fileId: content.file_id,
        },
      };
    }
    return undefined;
  }

  private convertToResponseInputTextContent(part: TextPart): ResponseInputItem.Message['content'][number] {
    return { type: 'input_text', text: part.text };
  }

  private convertToResponseInputImageContent(
    part: ImagePart,
  ): ResponseInputItem.Message['content'][number] | undefined {
    if (part.image.sourceType === 'url') {
      return { type: 'input_image', detail: 'auto', image_url: part.image.url };
    }
    if (part.image.sourceType === 'base64') {
      return {
        type: 'input_image',
        detail: 'auto',
        image_url: `data:${part.image.mimeType};base64,${part.image.data}`,
      };
    }
    if (part.image.sourceType === 'file_id') {
      return { type: 'input_image', detail: 'auto', file_id: part.image.fileId };
    }
    return undefined;
  }

  private convertToResponseInputContentPart(
    part: ContentPart,
  ): ResponseInputItem.Message['content'][number] | undefined {
    if (part.type === 'text') {
      return this.convertToResponseInputTextContent(part as TextPart);
    }
    if (part.type === 'image') {
      return this.convertToResponseInputImageContent(part as ImagePart);
    }
    return undefined;
  }

  private convertToResponseOutputTextContent(
    part: TextPart,
  ): ResponseOutputMessage['content'][number] {
    return {
      type: 'output_text',
      text: part.text,
      annotations: [],
    };
  }

  private convertToResponseOutputRefusalContent(
    part: RefusalPart,
  ): ResponseOutputMessage['content'][number] {
    return {
      type: 'refusal',
      refusal: part.reason,
    };
  }

  // private convertThinkingPartToResponseReasoningItem(
  //   part: ThinkingPart,
  // ): ResponseReasoningItem {
  //   return {
  //     type: 'reasoning',
  //     id: this.createSyntheticId('reasoning'),
  //     summary: [],
  //     content: [{ type: 'reasoning_text', text: part.content }],
  //     encrypted_content: part.signature,
  //   };
  // }

  private convertToResponseOutputContentPart(
    part: ContentPart,
  ): ResponseOutputMessage['content'][number] | null {
    if (part.type === 'text') {
      return this.convertToResponseOutputTextContent(part as TextPart);
    }
    if (part.type === 'refusal') {
      return this.convertToResponseOutputRefusalContent(part as RefusalPart);
    }
    return null;
  }

  private convertToResponseFunctionCallOutput(
    toolResult: ToolResultPart,
  ): ResponseInputItem.FunctionCallOutput {
    return {
      type: 'function_call_output',
      call_id: toolResult.id,
      output: toolResult.result,
      status: 'completed',
    } as ResponseInputItem.FunctionCallOutput;
  }

  private getFromMessage(rawMsg: ResponseApiMessage): SessionMessage {
    const parts = rawMsg.content.flatMap((item) => this.convertResponseInputItemToParts(item));
    
    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: rawMsg.role,
      content: parts.length === 0 ? '(message contains unsupported content)' : parts,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  convertFromRawMessage(rawMsg: ResponseApiMessage): SessionMessage {
    const message = this.getFromMessage(rawMsg);
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
      const converted = this.convertToResponseInputContentPart(part);
      if (converted) {
        content.push(converted);
      }
    }

    if (content.length === 0) {
      throw new InvalidMessageError(
        'User message must include at least one supported content part',
      );
    }

    return {
      type: 'message',
      role: 'user',
      content,
    };
  }

  private getAssistantMessage(
    msg: SessionMessage,
  ): ResponseInputItem | ResponseOutputMessage | Array<ResponseInputItem | ResponseOutputMessage> {
    if (!(msg.content instanceof Array)) {
      return {
        type: 'message',
        id: msg.messageId,
        role: 'assistant',
        status: 'completed',
        content: [{ type: 'output_text', text: msg.content ?? '', annotations: [] }],
      } as ResponseOutputMessage;
    }

    const items: Array<ResponseInputItem | ResponseOutputMessage> = [];

    // ignore the reasoning data. It may useless to get it from the message generated by other LLM.
    // const reasoningParts = msg.content.filter((part) => part.type === 'thinking') as ThinkingPart[];
    // reasoningParts.forEach((part) => items.push(this.convertThinkingPartToResponseReasoningItem(part)));

    const toolCalls = msg.content
      .filter((part) => part.type === 'tool_call')
      .map((part) => part as ToolCallPart);
    for (const toolCall of toolCalls) {
      items.push({
        type: 'function_call',
        id: `${msg.messageId}_${toolCall.id}`,
        call_id: toolCall.id,
        name: toolCall.name,
        arguments: toolCall.arguments,
        status: 'completed',
      } as ResponseFunctionToolCall);
    }

    const messageContent = msg.content
      .map((part) => this.convertToResponseOutputContentPart(part))
      .filter((item): item is ResponseOutputMessage['content'][number] => item !== null);

    if (messageContent.length > 0) {
      items.push({
        type: 'message',
        id: msg.messageId,
        role: 'assistant',
        status: 'completed',
        content: messageContent,
      } as ResponseOutputMessage);
    }

    return items.length === 1 ? items[0] : items;
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

    const rawMessages: ResponseInputItem[] = toolResults.map((toolResult) =>
      this.convertToResponseFunctionCallOutput(toolResult),
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

  convertToRawMessage(msg: SessionMessage): ResponseApiMessage {
    const rawItems = (() => {
      switch (msg.role) {
        case 'assistant':
          return [this.getAssistantMessage(msg)];
        case 'user':
          return [this.getUserMessage(msg)];
        case 'tool': {
          const toolRaw = this.getToolMessage(msg);
          return Array.isArray(toolRaw) ? toolRaw : [toolRaw];
        }
        default:
          throw new InvalidMessageError(`Unsupported message role: ${msg.role}`);
      }
    })();

    const responseApiMessage: ResponseApiMessage = {
      messageId: msg.messageId,
      type: 'response_api_message',
      role: msg.role,
      content: rawItems.flat(),
      metadata: msg.metadata,
    };

    return this.dialectResolver
      ? this.dialectResolver.manipulateRawMessage(responseApiMessage, msg)
      : responseApiMessage;
  }

  private getToolChoice(
    options: ModelOptions,
  ): ToolChoiceOptions | 'required' | 'none' | undefined {
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

  private getReasoningEffort(
    thinkEffort?: ModelOptions['thinkEffort'],
  ): 'none' | 'low' | 'medium' | 'high' | 'xhigh' | undefined {
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

  private getResponseMessage(response: Response): Omit<ResponseApiMessage, 'messageId'> {
    const supportedOutputItems: ResponseInputItem[] = [];
    for (const item of response.output) {
      if (item.type === 'function_call' || item.type === 'message' || item.type === 'reasoning') {
        supportedOutputItems.push(item as ResponseInputItem);
      }
    }

    const metadata: Record<string, unknown> = { 
      pixiagent_response_id: response.id,
      pixiagent_response_error: response.error,
      pixiagent_response_incomplete_details: response.incomplete_details,
      pixiagent_response_status: response.status,
    };

    if (supportedOutputItems.length > 0) {
      return {
        type: 'response_api_message',
        role: 'assistant',
        content: supportedOutputItems,
        metadata,
      };
    }

    return {
      type: 'response_api_message',
      role: 'assistant',
      content: [
        {
          type: 'message',
          id: response.id,
          role: 'assistant',
          status: 'completed',
          content: [{ type: 'output_text', text: response.output_text ?? '', annotations: [] }],
        } as ResponseOutputMessage,
      ],
      metadata,
    };
  }

  private getRefusal(responseMessage: Omit<ResponseApiMessage, 'messageId'>): string | undefined {
    for (const item of responseMessage.content) {
      if (item.type === 'message' && item.role === 'assistant' && Array.isArray(item.content)) {
        const refusal = item.content.find((content) => content.type === 'refusal');
        if (refusal?.type === 'refusal') {
          return refusal.refusal;
        }
      }
    }
    return undefined;
  }

  private buildParams(
    options: ModelOptions,
    messages: Array<ResponseApiMessage>,
  ): ResponseCreateParamsStreaming {
    const params: ResponseCreateParamsStreaming = {
      model: options.model,
      input: messages.flatMap((message) => message.content),
      instructions: options.systemPrompt,
      max_output_tokens: options.maxTokens,
      temperature: options.temperature,
      metadata: options.metadata,
      parallel_tool_calls: options.parallelToolCalls,
      tool_choice: this.getToolChoice(options),
      store: false,
      stream: true,
      reasoning: options.thinkEffort ? {
        effort: this.getReasoningEffort(options.thinkEffort),
        summary: 'auto',
      } : undefined,
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
    params: ResponseCreateParamsStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ResponseApiMessage, 'messageId'>>> {
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
      .stream(params, this.getStreamRequestOptions(requestOptions))
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
    params: ResponseCreateParamsStreaming,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ResponseApiMessage, 'messageId'>>> {
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
          const deltaEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.reasoning_text.delta' }
          >;
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
          const deltaEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.reasoning_summary_text.delta' }
          >;
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
          const doneEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.reasoning_text.done' }
          >;
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
          const doneEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.reasoning_summary_text.done' }
          >;
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
          const deltaEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.output_text.delta' }
          >;
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
          const doneEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.output_text.done' }
          >;
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
          const toolEvent = event as Extract<
            ResponseStreamEvent,
            { type: 'response.function_call_arguments.done' }
          >;
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
    messages: Array<ResponseApiMessage>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<ModelResponse<Omit<ResponseApiMessage, 'messageId'>>> {
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
