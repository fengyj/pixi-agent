import type { Logger } from 'pino';

import { getTracer, Helpers } from './observation';
import type { Span } from '@opentelemetry/api';
import { ErrorGuards, PixiAgentErrorBuilder } from './errors';
import { AgentMessageChunkEventCallbacks } from './event';
import {
  InternalMessage,
  ModelStopReasons,
  RawMessageType,
  SessionMessage,
  ToolCallPart,
  ToolResultPart,
  UsageStats,
} from './message';
import { SessionThread } from './session';
import { ContentParts } from './utils/contentpart';
import * as Transport from './transports';
import { ModelOptions, ProviderTransport } from './transports';
import { ToolRegistry } from './tools';
import type { AgentStopReason, PixiAgentExecutionResult, PixiAgentOptions } from './agent';

const trace = getTracer('pixiagent.agent');
const { withSpan, retry } = Helpers;

export class AgentExecutionLoop {
  private readonly transportCache = new Map<string, ProviderTransport<RawMessageType>>();
  private readonly convertedMessagesCache = new Map<string, RawMessageType | RawMessageType[]>();
  private _transport: ProviderTransport<RawMessageType>;

  constructor(
    private readonly sessionThread: SessionThread,
    private readonly toolRegistry: ToolRegistry,
    private readonly agentOptions: PixiAgentOptions,
    private readonly pendingMessages: Array<
      Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' }
    >,
    private readonly logger: Logger,
  ) {
    this._transport = AgentExecutionLoop.getTransport(sessionThread.threadInfo.modelOptions);
  }

  public async execute(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' },
    abortSignal: AbortSignal,
  ): Promise<PixiAgentExecutionResult> {
    const usage = UsageStats.empty();
    try {
      return await withSpan(
        'agent_execution',
        async () => {
          const transport = this.getTransport(modelOptions);
          const requestInternalMsg = this.appendRequestToThread(modelOptions, input);

          await this.agentOptions.eventEmitter.executionState.beforeModelRequest(
            this.sessionThread.session.sessionId,
            this.sessionThread.threadInfo.threadId,
            requestInternalMsg.rawMessage as SessionMessage,
          );

          const historyMessages = this.getHistoryMessagesForTransport(modelOptions, transport);
          const stopReason = await this.runExecutionLoop(
            transport,
            modelOptions,
            historyMessages,
            usage,
            abortSignal,
          );

          if (stopReason === 'end_turn') {
            this.logger.info('The agent execution is completed');
          }

          await this.agentOptions.eventEmitter.executionState.executionFinished(
            this.sessionThread.session.sessionId,
            this.sessionThread.threadInfo.threadId,
            stopReason,
            UsageStats.isEmpty(usage) ? undefined : usage,
          );

          return {
            action: 'execution',
            stopReason: stopReason,
            messages: input.messageId ? this.getNewMessages(input.messageId) : [],
          };
        },
        {
          tracer: trace,
          attrs: this.getExecuteSpanAttributes(),
          isExpectedError: (error) => this.isInterruptAbortError(error, abortSignal),
        },
      );
    } catch (error) {
      if (this.isInterruptAbortError(error, abortSignal)) {
        this.logger.warn(
          { cancel_reason: abortSignal.reason },
          'The agent execution is interrupted',
        );

        await this.agentOptions.eventEmitter.executionState
          .executionFinished(
            this.sessionThread.session.sessionId,
            this.sessionThread.threadInfo.threadId,
            'cancelled',
            UsageStats.isEmpty(usage) ? undefined : usage,
            error as Error,
          )
          .catch((emitError) => {
            this.logger.error(
              { error: emitError },
              'An error occurred while emitting executionFinished event',
            );
          });

        return {
          action: 'execution',
          stopReason: 'cancelled',
          messages: input.messageId ? this.getNewMessages(input.messageId) : [],
        };
      }
      this.logger.error({ error }, 'An error occurred during execution');
      await this.agentOptions.eventEmitter.executionState
        .executionFinished(
          this.sessionThread.session.sessionId,
          this.sessionThread.threadInfo.threadId,
          'cancelled',
          UsageStats.isEmpty(usage) ? undefined : usage,
          error as Error,
        )
        .catch((emitError) => {
          this.logger.error(
            { error: emitError },
            'An error occurred while emitting executionFinished event',
          );
        });
      throw error;
    }
  }

  private async runExecutionLoop(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    usage: UsageStats,
    abortSignal: AbortSignal,
  ): Promise<AgentStopReason> {
    let iterations = 0;
    while (abortSignal.aborted === false) {
      if (this.hasReachedMaxIterations(iterations)) {
        this.logger.warn(
          {
            iteration_times: iterations,
          },
          'The agent has reached the maximum number of iterations, stopping execution',
        );
        return 'max_turn_requests';
      }

      const response = await this.executeIteration(
        transport,
        modelOptions,
        historyMessages,
        iterations,
        abortSignal,
      );
      UsageStats.sum(usage, response.modelResponseInfo?.usage);

      await this.agentOptions.eventEmitter.executionState.iterationCompleted(
        this.sessionThread.session.sessionId,
        this.sessionThread.threadInfo.threadId,
        iterations,
      );

      iterations++;

      const stopReason = this.getLoopStopReason(response, modelOptions);
      if (stopReason) return stopReason;
    }

    return 'cancelled';
  }

  private async executeIteration(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    iteration: number,
    abortSignal: AbortSignal,
  ): Promise<RawMessageType> {
    return await withSpan(
      'agent_iteration',
      async () => {
        const response = await this.executeLLMRequest(
          transport,
          modelOptions,
          historyMessages,
          abortSignal,
        );

        if (response.modelResponseInfo?.stopReason === ModelStopReasons.TOOL_CALL) {
          await this.executeToolCallRequest(transport, modelOptions, historyMessages, abortSignal);
        }

        return response;
      },
      {
        tracer: trace,
        attrs: { 'agent.loop.iteration': iteration },
        isExpectedError: (error: unknown) => this.isInterruptAbortError(error, abortSignal),
      },
    );
  }

  private async executeLLMRequest(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    abortSignal: AbortSignal,
  ): Promise<RawMessageType> {
    return await withSpan(
      `chat ${modelOptions.model}`,
      async (span: Span) => {
        const createdAt = new Date().toISOString();
        const streamCallbacks = AgentMessageChunkEventCallbacks.create(
          this.agentOptions.eventEmitter,
          this.sessionThread.session.sessionId,
          this.sessionThread.threadInfo.threadId,
          'assistant',
          this.sessionThread.threadInfo.headMessageId ?? null,
        );

        const response = await retry(
          'chat.generate.retry',
          async () => {
            return await transport.generate(modelOptions, historyMessages, streamCallbacks, {
              signal: abortSignal,
              timeout: this.agentOptions.modelRequestTimeout,
            });
          },
          {
            span,
            maxAttempts: this.getLLMRequestMaxAttempts(),
            signal: abortSignal,
            attrs: {
              'gen_ai.operation.name': 'chat',
              'gen_ai.request.model': modelOptions.model,
            },
            isExpectedError: (error: unknown) => this.isInterruptAbortError(error, abortSignal),
          },
        );

        const responseInternalMsg = this.sessionThread.addMessage(
          'assistant',
          response,
          modelOptions,
          response.modelResponseInfo?.usage,
          createdAt,
        );
        const respSessionMsg = transport.convertFromRawMessage(
          responseInternalMsg.rawMessage as RawMessageType,
        );
        historyMessages.push(responseInternalMsg.rawMessage as RawMessageType);

        this.logger.info(
          AgentExecutionLoop.getMessageLogData(
            modelOptions,
            respSessionMsg,
            responseInternalMsg,
            response,
          ),
          'Response from the LLM',
        );
        this.addResponseSpanAttributes(span, modelOptions, response, responseInternalMsg);
        await streamCallbacks.onFinish(respSessionMsg);
        await this.agentOptions.eventEmitter.executionState.afterModelResponse(
          this.sessionThread.session.sessionId,
          this.sessionThread.threadInfo.threadId,
          respSessionMsg,
        );

        return responseInternalMsg.rawMessage as RawMessageType;
      },
      {
        tracer: trace,
        attrs: this.getRequestSpanAttributes(modelOptions),
        isExpectedError: (error: unknown) => this.isInterruptAbortError(error, abortSignal),
      },
    );
  }

  private async executeToolCallRequest(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    abortSignal: AbortSignal,
  ): Promise<void> {
    const payload = this.getToolCallPayload(transport, historyMessages);
    if (!payload) return;

    const { toolCalls } = payload;
    const isParallel = modelOptions.parallelToolCalls ?? true;
    const toolNames = toolCalls.map((tc) => tc.name);
    this.logger.debug({ toolNames, isParallel }, 'Tool calls start');

    await withSpan(
      'execute_tool_calls',
      async () => {
        const resultParts = await this.executeToolCalls(toolCalls, isParallel, abortSignal);
        await this.appendToolCallResults(transport, modelOptions, historyMessages, resultParts);
      },
      {
        tracer: trace,
        attrs: {
          'gen_ai.tool.names': toolNames.join(','),
          'agent.tool.parallel': isParallel,
        },
        isExpectedError: (error: unknown) => this.isInterruptAbortError(error, abortSignal),
      },
    );
  }

  private getToolCallPayload(
    transport: ProviderTransport<RawMessageType>,
    historyMessages: RawMessageType[],
  ): { sessionMessage: SessionMessage; toolCalls: ToolCallPart[] } | undefined {
    const sessionMessage = transport.convertFromRawMessage(
      historyMessages[historyMessages.length - 1],
    );
    if (!sessionMessage.content || typeof sessionMessage.content === 'string') return undefined;

    const toolCalls = sessionMessage.content.filter(
      (part) => part.type === 'tool_call',
    ) as ToolCallPart[];

    if (toolCalls.length === 0) return undefined;

    return { sessionMessage, toolCalls };
  }

  private async executeToolCalls(
    toolCalls: ToolCallPart[],
    isParallel: boolean,
    abortSignal: AbortSignal,
  ): Promise<ToolResultPart[]> {
    if (isParallel) {
      return await Promise.all(
        toolCalls.map(async (toolCall) => await this.executeSingleToolCall(toolCall, abortSignal)),
      );
    }

    const resultParts = [] as ToolResultPart[];
    for (const toolCall of toolCalls) {
      resultParts.push(await this.executeSingleToolCall(toolCall, abortSignal));
    }
    return resultParts;
  }

  private async executeSingleToolCall(
    toolCall: ToolCallPart,
    abortSignal: AbortSignal,
  ): Promise<ToolResultPart> {
    try {
      return await withSpan(
        `execute_tool ${toolCall.name}`,
        async () => {
          this.throwIfInterrupted(
            abortSignal,
            'The agent execution is interrupted, cannot execute tool call',
          );

          await this.agentOptions.eventEmitter.executionState.beforeToolCallRequest(
            this.sessionThread.session.sessionId,
            this.sessionThread.threadInfo.threadId,
            toolCall,
          );

          const result = await this.toolRegistry.execute(toolCall, {
            signal: abortSignal,
          });

          await this.agentOptions.eventEmitter.executionState.afterToolCallResponse(
            this.sessionThread.session.sessionId,
            this.sessionThread.threadInfo.threadId,
            toolCall,
            result,
          );

          return result;
        },
        {
          tracer: trace,
          attrs: {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': toolCall.name,
            'gen_ai.tool.call.id': toolCall.id,
          },
          isExpectedError: (error: unknown) => this.isInterruptAbortError(error, abortSignal),
        },
      );
    } catch (error) {
      const result = this.toToolCallErrorResult(toolCall, error);

      await this.agentOptions.eventEmitter.executionState
        .afterToolCallResponse(
          this.sessionThread.session.sessionId,
          this.sessionThread.threadInfo.threadId,
          toolCall,
          result,
        )
        .catch((emitError) => {
          this.logger.error(
            { error: emitError },
            'An error occurred while emitting afterToolCallResponse event',
          );
        });

      return result;
    }
  }

  private toToolCallErrorResult(toolCall: ToolCallPart, error: unknown): ToolResultPart {
    const message = error instanceof Error ? error.message : String(error);
    return {
      type: 'tool_result',
      id: toolCall.id,
      result: JSON.stringify({ error: message }),
      name: toolCall.name,
      isError: true,
    } as ToolResultPart;
  }

  private async appendToolCallResults(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    resultParts: ToolResultPart[],
  ): Promise<void> {
    const resultMsg = {
      type: 'session_message',
      role: 'tool',
      content:
        this.pendingMessages.length === 0
          ? resultParts
          : this.pendingMessages.map((msg) => msg.content).reduce(ContentParts.concat, resultParts),
    } as Omit<SessionMessage, 'messageId'>;

    if (this.pendingMessages.length > 0) {
      this.logger.debug('Appending pending messages content to tool call result');
      this.pendingMessages.length = 0;
    }

    const resultInternalMsg = this.sessionThread.addMessage(
      resultMsg.role,
      resultMsg,
      modelOptions,
    );

    await this.agentOptions.eventEmitter.executionState.toolCallsFinished(
      this.sessionThread.session.sessionId,
      this.sessionThread.threadInfo.threadId,
      resultInternalMsg.rawMessage as SessionMessage,
    );

    const rawResultMessages = this.toRawMessageArray(
      transport.convertToRawMessage(resultInternalMsg.rawMessage as SessionMessage),
    );
    historyMessages.push(...rawResultMessages);
    this.logger.info(
      AgentExecutionLoop.getMessageLogData(modelOptions, resultMsg, resultInternalMsg),
      'Tool calls completed',
    );
  }

  private hasReachedMaxIterations(iterations: number): boolean {
    return (
      this.agentOptions.maxIterations !== undefined && iterations >= this.agentOptions.maxIterations
    );
  }

  private getLoopStopReason(
    response: RawMessageType,
    modelOptions: ModelOptions,
  ): AgentStopReason | undefined {
    const stopReason = response.modelResponseInfo?.stopReason;
    if (stopReason === ModelStopReasons.TOOL_CALL) {
      return undefined;
    }
    if (stopReason === ModelStopReasons.STOP) {
      return 'end_turn';
    }
    if (stopReason === ModelStopReasons.MAX_TOKENS) {
      this.logger.warn(
        { max_tokens: modelOptions.maxTokens },
        'The agent execution is stopped due to reaching the maximum token limit',
      );
      return 'max_tokens';
    }
    if (stopReason === ModelStopReasons.REFUSAL) {
      const refusal = response.modelResponseInfo?.refusal;
      this.logger.warn({ refusal }, 'The agent execution is stopped due to refusal');
      return 'refusal';
    }
    if (stopReason === ModelStopReasons.CANCELLED || stopReason === ModelStopReasons.TIMEOUT) {
      return 'cancelled';
    }
    return 'cancelled';
  }

  private getRequestSpanAttributes(modelOptions: ModelOptions): Record<string, string | number> {
    const attrs: Record<string, string | number> = {
      'gen_ai.operation.name': 'chat',
      'gen_ai.request.model': modelOptions.model,
      'agent.api_mode': modelOptions.apiMode!,
    };
    if (modelOptions.maxTokens !== undefined) {
      attrs['gen_ai.request.max_tokens'] = modelOptions.maxTokens;
    }
    if (modelOptions.baseUrl) {
      attrs['server.address'] = modelOptions.baseUrl;
    }
    return attrs;
  }

  private addResponseSpanAttributes(
    span: Span,
    modelOptions: ModelOptions,
    response: Omit<RawMessageType, 'messageId'>,
    responseInternalMsg: InternalMessage,
  ): void {
    const modelRespInfo = response.modelResponseInfo;
    span.setAttribute('agent.request.internal_id', responseInternalMsg.previousMessageId!);
    span.setAttribute('agent.response.internal_id', responseInternalMsg.internalMessageId);
    if (modelRespInfo?.responseId)
      span.setAttribute('gen_ai.response.id', modelRespInfo.responseId);
    if (modelRespInfo?.stopReason)
      span.setAttribute('gen_ai.response.finish_reasons', [modelRespInfo.stopReason]);
    if (modelRespInfo?.responseModel)
      span.setAttribute('gen_ai.response.model', modelRespInfo.responseModel);
    if (modelRespInfo?.usage) {
      span.setAttribute('gen_ai.usage.total_tokens', modelRespInfo.usage.totalTokens);
      span.setAttribute('gen_ai.usage.input_tokens', modelRespInfo.usage.inputTokens);
      span.setAttribute('gen_ai.usage.output_tokens', modelRespInfo.usage.outputTokens);
      if (modelRespInfo.usage.cacheReadTokens !== undefined) {
        span.setAttribute(
          'gen_ai.usage.cache_read.input_tokens',
          modelRespInfo.usage.cacheReadTokens!,
        );
      }
      if (modelRespInfo.usage.cacheCreatedTokens !== undefined) {
        span.setAttribute(
          'gen_ai.usage.cache_creation.input_tokens',
          modelRespInfo.usage.cacheCreatedTokens!,
        );
      }
      if (modelRespInfo.usage.reasoningTokens !== undefined) {
        span.setAttribute(
          'gen_ai.usage.reasoning.output_tokens',
          modelRespInfo.usage.reasoningTokens!,
        );
      }
    }
    if (modelOptions.metadata) {
      span.setAttribute('agent.request.metadata', JSON.stringify(modelOptions.metadata));
    }
  }

  private getExecuteSpanAttributes(): Record<string, string> {
    return {
      'session.id': this.sessionThread.session.sessionId,
      'thread.id': this.sessionThread.threadInfo.threadId,
      'gen_ai.conversation.id': this.sessionThread.threadInfo.threadId,
      ...(this.sessionThread.session.parentSessionId
        ? { 'session.parent_id': this.sessionThread.session.parentSessionId }
        : {}),
    };
  }

  private appendRequestToThread(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId'> & { messageId?: string },
  ): InternalMessage {
    const requestInternalMsg = this.sessionThread.addMessage(input.role, input, modelOptions);
    this.logger.info(
      AgentExecutionLoop.getMessageLogData(modelOptions, input, requestInternalMsg),
      'Request to the LLM',
    );
    return requestInternalMsg;
  }

  private getHistoryMessagesForTransport(
    modelOptions: ModelOptions,
    transport: ProviderTransport<RawMessageType>,
  ): RawMessageType[] {
    const messages: RawMessageType[] = [];
    for (const msg of this.sessionThread.threadMessages) {
      const isAssistantMessage = msg.role === 'assistant';
      if (
        modelOptions.apiMode !== msg.apiMode ||
        modelOptions.baseUrl !== msg.baseUrl ||
        !isAssistantMessage
      ) {
        if (this.convertedMessagesCache.has(msg.internalMessageId)) {
          const cached = this.convertedMessagesCache.get(msg.internalMessageId)!;
          if (msg.role === 'tool') messages.push(...this.toRawMessageArray(cached));
          else messages.push(this.getSingleRawMessage(cached, msg.role, 'history cache'));
          continue;
        }
        const tKey = `${msg.apiMode}-${msg.baseUrl ?? ''}`;
        let t = this.transportCache.get(tKey);
        if (!t) {
          t = AgentExecutionLoop.getTransport({
            model: msg.model,
            apiMode: msg.apiMode,
            baseUrl: msg.baseUrl,
          });
          this.transportCache.set(tKey, t);
        }
        const sessionMsg = isAssistantMessage
          ? this.convertFromRawMessage(t, msg.rawMessage as RawMessageType)
          : (msg.rawMessage as SessionMessage);
        const converted = transport.convertToRawMessage(sessionMsg);
        this.convertedMessagesCache.set(msg.internalMessageId, converted);
        if (msg.role === 'tool') messages.push(...this.toRawMessageArray(converted));
        else messages.push(this.getSingleRawMessage(converted, msg.role, 'history conversion'));
      } else {
        messages.push(msg.rawMessage as RawMessageType);
      }
    }
    return messages;
  }

  private convertFromRawMessage(
    transport: ProviderTransport<RawMessageType>,
    rawMessage: RawMessageType,
  ): SessionMessage {
    const msg = transport.convertFromRawMessage(rawMessage);
    if (Array.isArray(msg.content) && msg.content.some((part) => part.type === 'thinking')) {
      msg.content = msg.content.filter((part) => part.type !== 'thinking');
    }
    return msg;
  }

  private toRawMessageArray(rawMessage: RawMessageType | RawMessageType[]): RawMessageType[] {
    return Array.isArray(rawMessage) ? rawMessage : [rawMessage];
  }

  private getSingleRawMessage(
    rawMessage: RawMessageType | RawMessageType[],
    role: SessionMessage['role'],
    stage: 'request' | 'history cache' | 'history conversion',
  ): RawMessageType {
    if (Array.isArray(rawMessage)) {
      throw PixiAgentErrorBuilder.invalidMessage(
        `Expected a single raw message for ${role} role at ${stage}, but got an array`,
        role,
      );
    }
    return rawMessage;
  }

  private getTransport(options: ModelOptions): ProviderTransport<RawMessageType> {
    const apiModeOrBaseUrlChanged =
      this.sessionThread.threadInfo.modelOptions.apiMode !== options.apiMode ||
      this.sessionThread.threadInfo.modelOptions.baseUrl !== options.baseUrl;
    const apiKeyChanged = this.sessionThread.threadInfo.modelOptions.apiKey !== options.apiKey;

    if (apiModeOrBaseUrlChanged || apiKeyChanged) {
      const transport = AgentExecutionLoop.getTransport(options, apiKeyChanged, this._transport);
      if (apiModeOrBaseUrlChanged) this.convertedMessagesCache.clear();
      this._transport = transport;
    }
    this.sessionThread.threadInfo.modelOptions = options;

    return this._transport;
  }

  private static getTransport(
    modelOptions: ModelOptions,
    apiKeyChanged = false,
    transport?: ProviderTransport<RawMessageType>,
  ): ProviderTransport<RawMessageType> {
    const dialectResolver = modelOptions.baseUrl
      ? Transport.GlobalDialectResolverRegistry.resolveDialect(
          modelOptions.model,
          modelOptions.baseUrl,
        )
      : undefined;

    if (
      transport &&
      !apiKeyChanged &&
      transport.apiMode === modelOptions.apiMode &&
      transport.dialectResolver === dialectResolver
    ) {
      return transport;
    }

    return Transport.getTransport(
      modelOptions.apiMode!,
      modelOptions.baseUrl,
      modelOptions.apiKey,
      dialectResolver,
    );
  }

  private getLLMRequestMaxAttempts(): number {
    const retries = this.agentOptions.maxModelRequestRetries;
    if (retries === undefined || retries <= 0) return 1;
    return Math.floor(retries) + 1;
  }

  private throwIfInterrupted(abortSignal: AbortSignal, defaultReason: string): void {
    if (!abortSignal.aborted) return;
    const signalReason = abortSignal.reason;
    if (signalReason instanceof Error) {
      throw signalReason;
    }
    throw PixiAgentErrorBuilder.agentInterrupted(defaultReason);
  }

  private isInterruptAbortError(error: unknown, abortSignal: AbortSignal): boolean {
    if (ErrorGuards.isLikelyAbortError(error)) {
      return true;
    }

    if (!abortSignal.aborted) {
      return false;
    }

    const signalReason = abortSignal.reason;
    if (error === signalReason) {
      return true;
    }

    return error instanceof Error && ErrorGuards.isLikelyAbortError(error);
  }

  private getNewMessages(userMsgId: string): SessionMessage[] {
    const newMessages: SessionMessage[] = [];

    let found = false;
    for (const msg of this.sessionThread.threadMessages) {
      if (msg.rawMessage && msg.rawMessage.messageId === userMsgId) {
        found = true;
      }
      if (found === false) continue;
      const sessionMsg =
        msg.rawMessage.type === 'session_message'
          ? (msg.rawMessage as SessionMessage)
          : this._transport.convertFromRawMessage(msg.rawMessage as RawMessageType);
      newMessages.push(sessionMsg);
    }
    return newMessages;
  }

  private static getMessageLogData(
    modelOptions: ModelOptions,
    sessionMessage: Omit<SessionMessage, 'messageId'> & { messageId?: string },
    internalMessage: InternalMessage,
    response?: Omit<RawMessageType, 'messageId'>,
  ): Record<string, unknown> {
    const modelRespInfo = response?.modelResponseInfo;
    const respData = modelRespInfo
      ? {
          responseId: modelRespInfo.responseId,
          responseModel: modelRespInfo.responseModel,
          stopReason: modelRespInfo.stopReason,
          usage: modelRespInfo.usage,
        }
      : {};

    return {
      ...respData,
      model: modelOptions.model,
      apiMode: modelOptions.apiMode,
      baseUrl: modelOptions.baseUrl,
      internalMessageId: internalMessage.internalMessageId,
      prevInternalMessageId: internalMessage.previousMessageId,
      role: sessionMessage.role,
      name: sessionMessage.name,
      content: ContentParts.digest(sessionMessage.content),
      metadata: modelOptions.metadata,
    };
  }
}
