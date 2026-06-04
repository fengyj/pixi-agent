/**
 * The agent is an interruptable loop. the inputs should be queued unless it requests an interrupt.
 *
 *
 * The agent manager is used as a bridge between requests and agents. It returns the agent based on
 * the session id. If the agent is not existed, load it from storage. And for the agents which don't active
 * for a while, makes them hibernation (persists the data, and remove them from memory). the manager also handles
 * the session transfer.
 *
 */

import { SessionThread } from './session';
import * as Transport from './transports';
import { ProviderTransport, ModelOptions } from './transports';
import {
  ContentPart,
  InternalMessage,
  RawMessageType,
  SessionMessage,
  ToolCallPart,
  ToolResultPart,
  UsageStats,
} from './message';
import { ToolRegistry } from './tools/tool';
import type { AgentSpan } from './observation';
import {
  AgentConcurrentExecutionError,
  AgentInterruptedError,
  ApiModeResolutionError,
  InvalidMessageError,
} from './errors/types';
import { ModelResponse } from './transports/base';

import * as Observation from './observation';
import { AgentEventEmitter } from './event';
const trace = Observation.getTracer('pixiagent.agent');
const { withSpan, retry } = Observation.Helpers;

export class PixiAgent {
  private readonly logger: ReturnType<ReturnType<typeof Observation.getLogger>['child']>;
  private _transport: ProviderTransport<RawMessageType>;
  private convertedMessagesCache = new Map<string, RawMessageType | RawMessageType[]>();
  private transportCache = new Map<string, ProviderTransport<RawMessageType>>();
  private abortController = new AbortController();
  private isRunning = false;
  // todo: add event listener as parameter to expose the events.
  constructor(
    public readonly sessionThread: SessionThread,
    public readonly toolRegistry: ToolRegistry,
    public readonly options?: PixiAgentOptions,
  ) {
    sessionThread.threadInfo.modelOptions = PixiAgent.resolveApiModeAndBaseUrl(
      sessionThread.threadInfo.modelOptions,
    );
    this._transport = PixiAgent.getTransport(sessionThread.threadInfo.modelOptions);
    this.logger = Observation.getLogger('agent').child({
      sessionId: sessionThread.session.sessionId,
      threadId: sessionThread.threadInfo.threadId,
    });
  }

  /**
   * The execute function is the main loop of the agent.
   * It takes the input, and execute the tasks in the input.
   * The input is a pending message, which is added to the session thread's pending messages.
   * The agent will peek the pending messages, and execute them one by one.
   * @param modelOptions
   * @param input
   */
  public async execute(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' },
  ): Promise<PixiAgentExecutionResult> {
    modelOptions = PixiAgent.resolveApiModeAndBaseUrl(modelOptions);
    if (this.isRunning) {
      throw new AgentConcurrentExecutionError();
    }

    this.isRunning = true;
    const usage = UsageStats.empty();
    let userMessageId: string | undefined = undefined;
    try {
      return await withSpan(
        'agent_execution',
        async () => {
          const transport = this.getTransport(modelOptions);
          const requestInternalMsg = this.enqueueRequestMessage(modelOptions, input);

          this.options?.eventEmitter?.emit({
            eventType: 'execution_state_changed',
            newState: 'before_model_request',
            message: requestInternalMsg.rawMessage as SessionMessage,
          });

          userMessageId = requestInternalMsg.internalMessageId;
          const historyMessages = this.getHistoryMessagesForTransport(modelOptions, transport);
          const stopReason = await this.runExecutionLoop(
            transport,
            modelOptions,
            historyMessages,
            usage,
          );

          if (stopReason === 'end_turn') {
            this.logger.info('The agent execution is completed');
          }

          this.options?.eventEmitter?.emit({
            eventType: 'execution_state_changed',
            newState: stopReason === 'end_turn' ? 'completed' : 'incomplete',
            stopReason: stopReason,
            usage: UsageStats.isEmpty(usage) ? undefined : usage,
          });

          return {
            stopReason,
            usage: UsageStats.isEmpty(usage) ? undefined : usage,
            userMessageId: userMessageId,
            metadata: this.getExecutionMetadata(),
          };
        },
        {
          tracer: trace,
          attrs: this.getExecuteSpanAttributes(),
          isExpectedError: (error) => this.isInterruptAbortError(error),
        },
      );
    } catch (error) {
      if (this.isInterruptAbortError(error)) {
        this.logger.warn(
          { cancel_reason: this.abortController.signal.reason },
          'The agent execution is interrupted',
        );
        
          this.options?.eventEmitter?.emit({
            eventType: 'execution_state_changed',
            newState: 'incomplete',
            stopReason: 'cancelled',
            usage: UsageStats.isEmpty(usage) ? undefined : usage,
            error: error as Error,
          });

        return {
          stopReason: 'cancelled',
          userMessageId: userMessageId,
          usage: UsageStats.isEmpty(usage) ? undefined : usage,
          metadata: this.getExecutionMetadata(),
        };
      }
      this.logger.error({ error }, 'An error occurred during execution');
      throw error;
    } finally {
      this.isRunning = false;
      this.resetAbortControllerIfNeeded();
    }
  }

  public interrupt(reason?: string): void {
    if (!this.isRunning) {
      this.logger.debug('The agent is not running, no need to interrupt');
      return;
    }
    if (this.abortController.signal.aborted) {
      this.logger.debug('The agent is already interrupted');
      return;
    }
    reason = reason ?? 'User interrupted';
    this.logger.info({ reason }, 'Interrupting the agent execution');
    this.abortController.abort(new AgentInterruptedError(reason));
  }

  private async executeLLMRequest(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
  ): Promise<ModelResponse<RawMessageType>> {
    return await withSpan(
      `chat ${modelOptions.model}`,
      async (span) => {
        const createdAt = new Date().toISOString();

        const response = await retry(
          'chat.generate.retry',
          async () => {
            return await transport.generate(
              modelOptions,
              historyMessages,
              {}, // todo: stream callbacks
              {
                signal: this.abortController.signal,
                timeout: this.options?.modelRequestTimeout,
              },
            );
          },
          {
            span,
            maxAttempts: this.getLLMRequestMaxAttempts(),
            signal: this.abortController.signal,
            attrs: {
              'gen_ai.operation.name': 'chat',
              'gen_ai.request.model': modelOptions.model,
            },
            isExpectedError: (error) => this.isInterruptAbortError(error),
          },
        );

        const responseInternalMsg = this.sessionThread.addMessage(
          'assistant',
          response.responseMessage,
          modelOptions,
          response.usage,
          createdAt,
        );
        const respSessionMsg = transport.convertFromRawMessage(
          responseInternalMsg.rawMessage as RawMessageType,
        );
        historyMessages.push(responseInternalMsg.rawMessage as RawMessageType);

        this.logger.info(
          PixiAgent.getMessageLogData(modelOptions, respSessionMsg, responseInternalMsg, response),
          'Response from the LLM',
        );
        this.addResponseSpanAttributes(span, modelOptions, response, responseInternalMsg);

        this.options?.eventEmitter?.emit({
          eventType: 'execution_state_changed',
          newState: 'after_model_response',
          message: respSessionMsg,
          hasToolCall: response.stopReason === 'tool_call' || false,
          usage: response.usage,
        });

        return {
          ...response,
          responseMessage: {
            ...response.responseMessage,
            messageId: responseInternalMsg.rawMessage.messageId,
          } as RawMessageType,
        } as ModelResponse<RawMessageType>;
      },
      {
        tracer: trace,
        attrs: this.getRequestSpanAttributes(modelOptions),
        isExpectedError: (error) => this.isInterruptAbortError(error),
      },
    );
  }

  private async executeToolCallRequest(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
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
        const resultParts = await this.executeToolCalls(toolCalls, isParallel);
        this.appendToolCallResults(transport, modelOptions, historyMessages, resultParts);
      },
      {
        tracer: trace,
        attrs: {
          'gen_ai.tool.names': toolNames.join(','),
          'agent.tool.parallel': isParallel,
        },
        isExpectedError: (error) => this.isInterruptAbortError(error),
      },
    );
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

  private enqueueRequestMessage(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId'> & { messageId?: string },
  ): InternalMessage {
    const requestInternalMsg = this.sessionThread.addMessage(input.role, input, modelOptions);
    this.logger.info(
      PixiAgent.getMessageLogData(modelOptions, input, requestInternalMsg),
      'Request to the LLM',
    );
    return requestInternalMsg;
  }

  private async runExecutionLoop(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    usage: UsageStats,
  ): Promise<AgentStopReason> {
    let iterations = 0;
    while (this.abortController.signal.aborted === false) {
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
      );
      UsageStats.sum(usage, response.usage);

      this.options?.eventEmitter?.emit({
        eventType: 'execution_state_changed',
        newState: 'iteration_completed',
        iteration: iterations,
      });

      iterations++;

      const stopReason = this.getLoopStopReason(response, modelOptions);
      if (stopReason) return stopReason;
    }

    return 'cancelled';
  }

  private hasReachedMaxIterations(iterations: number): boolean {
    return this.options?.maxIterations !== undefined && iterations >= this.options.maxIterations;
  }

  private async executeIteration(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    iteration: number,
  ): Promise<ModelResponse<RawMessageType>> {
    return await withSpan(
      'agent_iteration',
      async () => {
        this.ensureMediaSources(modelOptions, transport, historyMessages);
        const response = await this.executeLLMRequest(transport, modelOptions, historyMessages);

        if (response.stopReason === 'tool_call') {
          await this.executeToolCallRequest(transport, modelOptions, historyMessages);
        }

        return response;
      },
      {
        tracer: trace,
        attrs: { 'agent.loop.iteration': iteration },
        isExpectedError: (error) => this.isInterruptAbortError(error),
      },
    );
  }

  /**
   * Ensure the media sources in the history messages are valid. This is used for the media sources
   * which are not embedded as base64 in the messages.
   * If the media source is expired (URL or file id), or the file id is not for the provider,
   * will refresh the URL or reupload the file to the current provider, and update the media source
   * in the message.
   * @param modelOptions
   * @param transport
   * @param historyMessages
   */
  private ensureMediaSources(
    modelOptions: ModelOptions,
    transport: ProviderTransport<RawMessageType>,
    historyMessages: RawMessageType[],
  ): void {
    // todo: if the media source in any message is expired, need to
    //       refresh the source, and reproduce the history messages
    if (!this.ensureMediaValid()) {
      // todo: raise event to handle the expired media
      historyMessages.splice(
        0,
        historyMessages.length,
        ...this.getHistoryMessagesForTransport(modelOptions, transport),
      );
    }
  }

  /**
   * Returns undefined when the reason is tool_call, otherwise returns the stop reason.
   * For the reasons which are not directly mapped to the agent stop reason,
   * will return 'cancelled'.
   * @param response
   * @param modelOptions
   * @returns
   */
  private getLoopStopReason(
    response: ModelResponse<RawMessageType>,
    modelOptions: ModelOptions,
  ): AgentStopReason | undefined {
    if (response.stopReason === 'tool_call') {
      return undefined;
    }

    if (response.stopReason === 'stop') {
      return 'end_turn';
    }

    if (response.stopReason === 'max_tokens') {
      this.logger.warn(
        {
          max_tokens: modelOptions.maxTokens,
        },
        'The agent execution is stopped due to reaching the maximum token limit',
      );
      return 'max_tokens';
    }

    if (response.stopReason === 'refusal') {
      const refusal = (response as { refusal?: string }).refusal;
      this.logger.warn(
        {
          refusal,
        },
        'The agent execution is stopped due to refusal',
      );
      return 'refusal';
    }

    if (response.stopReason === 'cancelled' || response.stopReason === 'timeout') {
      return 'cancelled';
    }

    return 'cancelled';
  }

  private getExecutionMetadata(): Record<string, unknown> {
    const metadata: Record<string, unknown> = {};
    const threadInfo = this.sessionThread.threadInfo;

    if (threadInfo.threadId) {
      metadata.PIXI_AGENT_THREAD_ID = threadInfo.threadId;
    }
    if (threadInfo.headMessageId) {
      metadata.PIXI_AGENT_HEAD_MESSAGE_ID = threadInfo.headMessageId;
    }
    if (threadInfo.rootMessageId) {
      metadata.PIXI_AGENT_ROOT_MESSAGE_ID = threadInfo.rootMessageId;
    }
    if (threadInfo.title) {
      metadata.PIXI_AGENT_TITLE = threadInfo.title;
    }

    return metadata;
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

  private getLLMRequestMaxAttempts(): number {
    const retries = this.options?.maxModelRequestRetries;
    if (retries === undefined || retries <= 0) return 1;
    return Math.floor(retries) + 1;
  }

  private addResponseSpanAttributes(
    span: AgentSpan,
    modelOptions: ModelOptions,
    response: ModelResponse<Omit<RawMessageType, 'messageId'>>,
    responseInternalMsg: InternalMessage,
  ): void {
    span.setAttribute('agent.request.internal_id', responseInternalMsg.previousMessageId!);
    span.setAttribute('agent.response.internal_id', responseInternalMsg.internalMessageId);
    if (response.responseId) span.setAttribute('gen_ai.response.id', response.responseId);
    if (response.stopReason)
      span.setAttribute('gen_ai.response.finish_reasons', [response.stopReason]);
    if (response.responseModel) span.setAttribute('gen_ai.response.model', response.responseModel);
    if (response.usage) {
      span.setAttribute('gen_ai.usage.total_tokens', response.usage.totalTokens);
      span.setAttribute('gen_ai.usage.input_tokens', response.usage.inputTokens);
      span.setAttribute('gen_ai.usage.output_tokens', response.usage.outputTokens);
      if (response.usage.cacheReadTokens !== undefined)
        span.setAttribute('gen_ai.usage.cache_read.input_tokens', response.usage.cacheReadTokens!);
      if (response.usage.cacheCreatedTokens !== undefined)
        span.setAttribute(
          'gen_ai.usage.cache_creation.input_tokens',
          response.usage.cacheCreatedTokens!,
        );
      if (response.usage.reasoningTokens !== undefined)
        span.setAttribute('gen_ai.usage.reasoning.output_tokens', response.usage.reasoningTokens!);
    }
    if (modelOptions.metadata) {
      span.setAttribute('agent.request.metadata', JSON.stringify(modelOptions.metadata));
    }
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
  ): Promise<ToolResultPart[]> {
    if (isParallel) {
      return await Promise.all(
        toolCalls.map(async (toolCall) => await this.executeSingleToolCall(toolCall)),
      );
    }

    const resultParts = [] as ToolResultPart[];
    for (const toolCall of toolCalls) {
      resultParts.push(await this.executeSingleToolCall(toolCall));
    }
    return resultParts;
  }

  private async executeSingleToolCall(toolCall: ToolCallPart): Promise<ToolResultPart> {
    // todo: implement toolcall options
    // todo: implement event callbacks
    try {
      return await withSpan(
        `execute_tool ${toolCall.name}`,
        async () => {
          this.throwIfInterrupted('The agent execution is interrupted, cannot execute tool call');

          this.options?.eventEmitter?.emit({
            eventType: 'execution_state_changed',
            newState: 'before_tool_call_request',
            toolCall: toolCall,
          });

          const result = await this.toolRegistry.execute(toolCall, {
            signal: this.abortController.signal,
          });

          this.options?.eventEmitter?.emit({
            eventType: 'execution_state_changed',
            newState: 'after_tool_call_response',
            toolCall: toolCall,
            toolResult: result,
          });

          return result;
        },
        {
          tracer: trace,
          attrs: {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': toolCall.name,
            'gen_ai.tool.call.id': toolCall.id,
          },
          isExpectedError: (error) => this.isInterruptAbortError(error),
        },
      );
    } catch (error) {
      const result = this.toToolCallErrorResult(toolCall, error);

      this.options?.eventEmitter?.emit({
        eventType: 'execution_state_changed',
        newState: 'after_tool_call_response',
        toolCall: toolCall,
        toolResult: result,
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

  private appendToolCallResults(
    transport: ProviderTransport<RawMessageType>,
    modelOptions: ModelOptions,
    historyMessages: RawMessageType[],
    resultParts: ToolResultPart[],
  ): void {
    const resultMsg = {
      type: 'session_message',
      role: 'tool',
      content: resultParts,
    } as Omit<SessionMessage, 'messageId'>;

    const resultInternalMsg = this.sessionThread.addMessage(
      resultMsg.role,
      resultMsg,
      modelOptions,
    );

    this.options?.eventEmitter?.emit({
      eventType: 'execution_state_changed',
      newState: 'tool_calls_finished',
      message: resultInternalMsg.rawMessage as SessionMessage,
    });

    const rawResultMessages = this.toRawMessageArray(
      transport.convertToRawMessage(resultInternalMsg.rawMessage as SessionMessage),
    );
    historyMessages.push(...rawResultMessages);
    this.convertedMessagesCache.set(resultInternalMsg.internalMessageId, rawResultMessages);
    this.logger.info(
      PixiAgent.getMessageLogData(modelOptions, resultMsg, resultInternalMsg),
      'Tool calls completed',
    );
  }

  private throwIfInterrupted(defaultReason: string): void {
    if (!this.abortController.signal.aborted) return;
    const signalReason = this.abortController.signal.reason;
    if (signalReason instanceof Error) {
      throw signalReason;
    }
    throw new AgentInterruptedError(defaultReason);
  }

  private isInterruptAbortError(error: unknown): boolean {
    if (error instanceof AgentInterruptedError) {
      return true;
    }

    if (!this.abortController.signal.aborted) {
      return false;
    }

    const signalReason = this.abortController.signal.reason;
    if (!(signalReason instanceof AgentInterruptedError)) {
      return false;
    }

    if (error === signalReason) {
      return true;
    }

    if (error instanceof Error) {
      const errorName = error.name.toLowerCase();
      const errorMessage = error.message.toLowerCase();
      return (
        errorName === 'aborterror' ||
        errorName.includes('abort') ||
        errorMessage.includes('request was aborted') ||
        errorMessage.includes('aborted')
      );
    }

    return false;
  }

  private resetAbortControllerIfNeeded(): void {
    if (this.abortController.signal.aborted) {
      this.abortController = new AbortController();
    }
  }

  /**
   * Get the history messages and if any of them was not in the same API mode as the current request,
   * convert it to the current transport's raw message format.
   * @param modelOptions
   * @returns
   */
  private getHistoryMessagesForTransport(
    modelOptions: ModelOptions,
    transport: ProviderTransport<RawMessageType>,
  ): RawMessageType[] {
    const messages = [] as RawMessageType[];
    for (const msg of this.sessionThread.threadMessages) {
      const isAssistantMessage = msg.role === 'assistant';
      if (
        modelOptions.apiMode !== msg.apiMode ||
        modelOptions.baseUrl !== msg.baseUrl ||
        !isAssistantMessage
      ) {
        if (this.convertedMessagesCache.has(msg.internalMessageId)) {
          const cached = this.convertedMessagesCache.get(msg.internalMessageId)!;
          if (msg.role === 'tool') {
            messages.push(...this.toRawMessageArray(cached));
          } else {
            messages.push(this.getSingleRawMessage(cached, msg.role, 'history cache'));
          }
          continue;
        }
        // not comparing the model here, because there is no dialect resolver relies on the model for now.
        const tKey = `${msg.apiMode}-${msg.baseUrl ?? ''}`;
        let t = this.transportCache.get(tKey);
        if (!t) {
          t = PixiAgent.getTransport({
            model: msg.model,
            apiMode: msg.apiMode,
            baseUrl: msg.baseUrl,
          });
          this.transportCache.set(tKey, t);
        }
        const sessionMsg = isAssistantMessage
          ? t.convertFromRawMessage(msg.rawMessage as RawMessageType)
          : (msg.rawMessage as SessionMessage);
        const converted = transport.convertToRawMessage(sessionMsg);
        this.convertedMessagesCache.set(msg.internalMessageId, converted);
        if (msg.role === 'tool') {
          messages.push(...this.toRawMessageArray(converted));
        } else {
          messages.push(this.getSingleRawMessage(converted, msg.role, 'history conversion'));
        }
      } else {
        messages.push(msg.rawMessage as RawMessageType);
      }
    }
    return messages;
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
      throw new InvalidMessageError(
        `Expected a single raw message for ${role} role at ${stage}, but got ${rawMessage.length}`,
      );
    }
    return rawMessage;
  }

  private getTransport(options: ModelOptions): ProviderTransport<RawMessageType> {
    // not comparing the model here, because there is no dialect resolver relies on the model for now.
    if (
      this.sessionThread.threadInfo.modelOptions.apiMode !== options.apiMode ||
      this.sessionThread.threadInfo.modelOptions.baseUrl !== options.baseUrl
    ) {
      const transport = PixiAgent.getTransport(options, this._transport);
      this.convertedMessagesCache.clear();
      this._transport = transport;
      this.sessionThread.threadInfo.modelOptions = options;
    }
    return this._transport;
  }

  /**
   *
   * @returns if any media source is expired, return false.
   */
  private ensureMediaValid(): boolean {
    const now = Date.now() + 1000 * 60 * 10; // 10 minutes buffer
    for (const msg of this.sessionThread.threadMessages) {
      if ((msg.rawMessage as SessionMessage)?.type === 'session_message') {
        const sessionMsg = msg.rawMessage as SessionMessage;
        if (!sessionMsg.content || typeof sessionMsg.content === 'string') continue;
        for (const part of sessionMsg.content) {
          const media = (() => {
            switch (part.type) {
              case 'image':
                return part.image;
              case 'document':
                return part.document;
              case 'audio':
                return part.audio;
              case 'video':
                return part.video;
              default:
                return undefined;
            }
          })();
          if (
            media !== undefined &&
            media.sourceType !== 'base64' &&
            media?.expireAt &&
            media.expireAt <= now
          ) {
            // the mediaInfo is session level (if it's thread level, need to be copied during the forking),
            // so, when the event is raised, to update all the expired media sources in all messages or just
            // the messages in the current thread, we leave it to the event handler to decide.

            // todo: raise event to handle the expired media
            this.convertedMessagesCache.clear();
            return false;
          }
        }
      }
    }
    return true;
  }

  private static getMessageLogData(
    modelOptions: ModelOptions,
    sessionMessage: Omit<SessionMessage, 'messageId'> & { messageId?: string },
    internalMessage: InternalMessage,
    response?: ModelResponse<Omit<RawMessageType, 'messageId'>>,
  ): Record<string, unknown> {
    const respData = response
      ? {
          responseId: response.responseId,
          responseModel: response.responseModel,
          stopReason: response.stopReason,
          usage: response.usage,
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
      content: ContentPart.digest(sessionMessage.content),
      metadata: modelOptions.metadata,
    };
  }

  private static resolveApiModeAndBaseUrl(modelOptions: ModelOptions): ModelOptions {
    const resolveResult = Transport.GlobalApiModeResolverRegistry.resolve(
      modelOptions.model,
      modelOptions.baseUrl,
      modelOptions.apiMode,
    );
    if (!resolveResult) {
      throw new ApiModeResolutionError(modelOptions.model, modelOptions.baseUrl);
    }
    const [baseUrl, apiMode] = resolveResult;
    if (modelOptions.apiMode !== apiMode || modelOptions.baseUrl !== baseUrl) {
      return {
        ...modelOptions,
        apiMode,
        baseUrl,
      };
    }
    return modelOptions;
  }

  private static getTransport(
    modelOptions: ModelOptions,
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
}

export interface PixiAgentOptions {
  /**
   * The timeout for the model request. If the model request takes longer than this time,
   * it will be aborted.
   * The unit is milliseconds.
   *
   * The tool calls timeout is defined in the ToolRegistry.
   */
  modelRequestTimeout?: number;
  /**
   * The maximum number of iterations for the agent to execute.
   * If the agent executes more than this number, it will stop.
   */
  maxIterations?: number;
  /**
   * The maximum number of retries for model requests.
   * If the model request fails more than this number, it will stop.
   * If its value is undefined, or less or equal to 0, it will not retry.
   *
   * Retry is only for retriable errors, such as network errors, or server errors (5xx).
   *
   * The retry times for tool call is defined in the ToolRegistry.
   */
  maxModelRequestRetries?: number;
  /**
   * The event emitter to emit the agent events.
   */
  eventEmitter?: AgentEventEmitter;
}

export type AgentStopReason =
  | 'end_turn'
  | 'max_turn_requests'
  | 'max_tokens'
  | 'refusal'
  | 'cancelled';

export interface PixiAgentExecutionResult {
  stopReason: AgentStopReason;
  usage?: UsageStats;
  userMessageId?: string;
  metadata?: Record<string, unknown>;
}
