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

import { SessionThread, PendingMessage } from './session';
import { Transport, ProviderTransport, ModelOptions } from './transports';
import {
  ContentPart,
  RawMessageType,
  SessionMessage,
  ToolCallPart,
  ToolResultPart,
} from './message';
import { ToolRegistry } from './tool';
import { Observation } from './observation';
import { SpanStatusCode } from '@opentelemetry/api';
import {
  AgentInterruptedError,
  ApiModeResolutionError,
  MaxIterationsExceededError,
} from './errors';

const logger = Observation.getLogger('agent');
const trace = Observation.getTracer('pixiagent.agent');

export class PixiAgent {
  private _transport: ProviderTransport<RawMessageType>;
  private convertedMessagesCache = new Map<string, RawMessageType>();
  private transportCache = new Map<string, ProviderTransport<RawMessageType>>();
  private abortController = new AbortController();
  private isRunning = false;
  private readonly logger: ReturnType<typeof logger.child>;
  // todo: add event listener as parameter to expose the events.
  constructor(
    public sessionThread: SessionThread,
    public toolRegistry: ToolRegistry,
    public options?: PixiAgentOptions,
  ) {
    sessionThread.threadInfo.modelOptions = PixiAgent.resolveApiModeAndBaseUrl(
      sessionThread.threadInfo.modelOptions,
    );
    this._transport = PixiAgent.getTransport(sessionThread.threadInfo.modelOptions);
    this.logger = logger.child({
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
   * @param input todo: add InterruptionMessage
   */
  public async execute(modelOptions: ModelOptions, input: Omit<PendingMessage, 'pendingMessageId'>): Promise<void> {
    const pendingMessage = this.sessionThread.addPendingMessage(input);
    this.logger.debug(
      {
        pendingMessage: {
          role: pendingMessage.role,
          pendingMessageId: pendingMessage.pendingMessageId,
          content: pendingMessage.content, // todo: use a function to summarize the content
          name: pendingMessage.name,
          refusal: pendingMessage.refusal,
        },
        pendingQueueLength: this.sessionThread.getPendingMessages().length,
      },
      'Received new input, added to pending messages',
    );
    /**
     * todo: there is an issue that the new modelOptions will be lost.
     */
    if (this.isRunning) {
      this.logger.warn('The agent is already running, the new input is queued');
      return;
    }

    modelOptions = PixiAgent.resolveApiModeAndBaseUrl(modelOptions);

    return await trace.startActiveSpan('agent_execution', async (span) => {
      span.setAttribute('session.id', this.sessionThread.session.sessionId);
      span.setAttribute('thread.id', this.sessionThread.threadInfo.threadId);
      span.setAttribute('gen_ai.conversation.id', this.sessionThread.threadInfo.threadId);
      if (this.sessionThread.session.parentSessionId)
        span.setAttribute('session.parent_id', this.sessionThread.session.parentSessionId);

      this.isRunning = true;

      if (this.abortController.signal.aborted) {
        this.abortController = new AbortController();
      }
      try {
        let iterations = 0;
        while (!this.abortController.signal.aborted) {
          if (this.sessionThread.getPendingMessages().length === 0) {
            break;
          }
          if (this.options?.maxIterations && iterations >= this.options.maxIterations) {
            this.logger.warn(
              {
                iteration_times: iterations,
              },
              'The agent has reached the maximum number of iterations, stopping execution',
            );
            throw new MaxIterationsExceededError(this.options.maxIterations);
          }
          await trace.startActiveSpan('agent_iteration', async (iterationSpan) => {
            iterationSpan.setAttribute('agent.loop.iteration', iterations);
            try {
              await this.consumePendingMessages(modelOptions);
            } catch (error) {
              iterationSpan.recordException(error as Error);
              iterationSpan.setStatus({ code: SpanStatusCode.ERROR, message: (error as Error).message });
              iterationSpan.setAttribute('error.type', (error as Error).name || 'Error');
              this.logger.error({ error }, 'An error occurred during consuming pending messages');
            } finally {
            iterationSpan.end();
            }
            // todo: a hook here can be used for session persistence.
          });
          iterations++;
        }
        if (this.abortController.signal.aborted) {
          this.logger.info('The agent execution is interrupted');
        } else {
          this.logger.debug('The agent execution is completed');
        }
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: (error as Error).message });
        span.setAttribute('error.type', (error as Error).name || 'Error');
        this.logger.error({ error }, 'An error occurred during execution');
      } finally {
        this.isRunning = false;
        span.end();
      }
    });
  }

  public interrupt(reason?: string): void {
    if (this.abortController.signal.aborted) {
      this.logger.debug('The agent is already interrupted');
      return;
    }
    reason = reason ?? 'User interrupted';
    this.logger.info({ reason }, 'Interrupting the agent execution');
    this.abortController.abort(new AgentInterruptedError(reason));
  }

  private async consumePendingMessages(modelOptions: ModelOptions): Promise<void> {
    const pendingMessages = PixiAgent.peekPendingMessagesToExecute(this.sessionThread);
    const sessionMessage = PixiAgent.convertPendingMessages(this.sessionThread, pendingMessages);

    if (
      !sessionMessage.refusal &&
      (!sessionMessage.content ||
        (typeof sessionMessage.content !== 'string' && sessionMessage.content.length === 0))
    ) {
      this.sessionThread.removePendingMessage(pendingMessages.map((msg) => msg.pendingMessageId));
      // todo: add a warning log here for the invalid pending message.
      pendingMessages.forEach((msg) => {
        this.logger.warn(
          {
            pendingMessage: {
              role: msg.role,
              pendingMessageId: msg.pendingMessageId,
              content: msg.content, // todo: use a function to summarize the content
              name: msg.name,
              refusal: msg.refusal,
            },
          },
          'The pending message is invalid, skipping execution',
        );
      });
      return;
    }
    pendingMessages.forEach((msg) => {
      this.logger.debug(
        {
          pendingMessage: {
            role: msg.role,
            pendingMessageId: msg.pendingMessageId,
            content: msg.content, // todo: use a function to summarize the content
            name: msg.name,
            refusal: msg.refusal,
          },
        },
        'Consuming pending message',
      );
    });

    switch (sessionMessage.role) {
      case 'user': // handle user message
      case 'tool': // handle tool result message
        await this.executeLLMRequest(modelOptions, sessionMessage);
        break;
      case 'assistant': // handle tool call and add to pending messages
        await this.executeToolCallRequest(modelOptions, sessionMessage);
        break;
      default:
        throw new Error(`Unknown message role ${sessionMessage.role}`);
    }
    this.sessionThread.threadInfo.modelOptions = modelOptions;
    this.sessionThread.removePendingMessage(pendingMessages.map((msg) => msg.pendingMessageId));
    pendingMessages.forEach((msg) => {
      this.logger.debug(
        {
          pendingMessage: {
            role: msg.role,
            pendingMessageId: msg.pendingMessageId,
            content: msg.content, // todo: use a function to summarize the content
            name: msg.name,
            refusal: msg.refusal,
          },
          pendingQueueLength: this.sessionThread.getPendingMessages().length,
        },
        'Finished consuming pending message, removed from pending messages',
      );
    });
  }

  private async executeLLMRequest(
    modelOptions: ModelOptions,
    sessionMessage: SessionMessage,
  ): Promise<void> {
    return await trace.startActiveSpan(`chat ${modelOptions.model}`, async (span) => {
      span.setAttribute('gen_ai.operation.name', 'chat');
      // todo: add gen_ai.provider.name attribute.
      span.setAttribute('gen_ai.request.model', modelOptions.model);
      if (modelOptions.maxTokens)
        span.setAttribute('gen_ai.request.max_tokens', modelOptions.maxTokens);
      span.setAttribute('agent.api_mode', modelOptions.apiMode!);
      if (modelOptions.baseUrl) span.setAttribute('server.address', modelOptions.baseUrl);

      try {
        const transport = this.getTransport(modelOptions);
        const historyMessages = this.getHistoryMessagesForTransport(modelOptions);
        const rawRequestMessage = transport.convertToRawMessage(sessionMessage);
        const createdAt = new Date().toISOString();

        const response = await transport.generate(
          modelOptions,
          [...historyMessages, rawRequestMessage],
          {}, // todo: stream callbacks
          {
            signal: this.abortController.signal,
            timeout: this.options?.modelRequestTimeout,
          },
        );

        // append the input message
        const requestMsg = this.sessionThread.addMessage(rawRequestMessage, modelOptions);
        this.logger.info(
          {
            model: modelOptions.model,
            apiMode: modelOptions.apiMode,
            baseUrl: modelOptions.baseUrl,
            inputMessage: {
              internalMessageId: requestMsg.internalMessageId,
              role: sessionMessage.role,
              name: sessionMessage.name,
              refusal: sessionMessage.refusal,
              content: sessionMessage.content, // todo: use a function to summarize the content
            },
            metadata: modelOptions.metadata,
          },
          'Request to the model',
        );

        const respSessionMsg = transport.convertFromRawMessage(response.responseMessage);
        if (
          !respSessionMsg.refusal &&
          respSessionMsg.content &&
          typeof respSessionMsg.content !== 'string'
        ) {
          const parts = respSessionMsg.content! as ContentPart[];
          const toolParts = parts.filter((part) => part.type === 'tool_call') as ContentPart[];
          if (toolParts.length > 0) {
            const pendingToolCallMessage = this.sessionThread.addPendingMessage({
              type: 'pending_message',
              role: 'assistant',
              name: sessionMessage.name,
              content: toolParts,
            });
            this.logger.debug(
              {
                pendingMessage: {
                  role: pendingToolCallMessage.role,
                  pendingMessageId: pendingToolCallMessage.pendingMessageId,
                  content: pendingToolCallMessage.content, // todo: use a function to summarize the content
                  name: pendingToolCallMessage.name,
                  refusal: pendingToolCallMessage.refusal,
                },
              },
              'Added to pending messages for tool calls',
            );
          }
        }

        // apend the response message
        const responseMsg = this.sessionThread.addMessage(
          response.responseMessage,
          modelOptions,
          response.usage,
          createdAt,
        );
        this.logger.info(
          {
            model: modelOptions.model,
            apiMode: modelOptions.apiMode,
            baseUrl: modelOptions.baseUrl,
            responseMessage: {
              internalMessageId: responseMsg.internalMessageId,
              rawMessageId: response.responseId,
              role: respSessionMsg.role,
              name: respSessionMsg.name,
              refusal: respSessionMsg.refusal,
              content: respSessionMsg.content, // todo: use a function to summarize the content
            },
            usage: response.usage,
            metadata: modelOptions.metadata,
          },
          'Response from the model',
        );
        span.setAttribute('agent.request.internal_id', requestMsg.internalMessageId);
        span.setAttribute('agent.response.internal_id', responseMsg.internalMessageId);
        if(response.responseId)
          span.setAttribute('gen_ai.response.id', response.responseId);
        if (response.stopReason)
          span.setAttribute('gen_ai.response.finish_reasons', [response.stopReason]);
        if (response.responseModel)
          span.setAttribute('gen_ai.response.model', response.responseModel);
        if (response.usage) {
          span.setAttribute('gen_ai.usage.total_tokens', response.usage.totalTokens);
          span.setAttribute('gen_ai.usage.input_tokens', response.usage.inputTokens);
          span.setAttribute('gen_ai.usage.output_tokens', response.usage.outputTokens);
          if (response.usage.cacheReadTokens !== undefined)
            span.setAttribute(
              'gen_ai.usage.cache_read.input_tokens',
              response.usage.cacheReadTokens!,
            );
          if (response.usage.cacheCreatedTokens !== undefined)
            span.setAttribute(
              'gen_ai.usage.cache_creation.input_tokens',
              response.usage.cacheCreatedTokens!,
            );
          if (response.usage.reasoningTokens !== undefined)
            span.setAttribute(
              'gen_ai.usage.reasoning.output_tokens',
              response.usage.reasoningTokens!,
            );
        }
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: (error as Error).message });
        span.setAttribute('error.type', (error as Error).name || 'Error');
        throw error;
      } finally {
        span.end();
      }
    });
  }

  private async executeToolCallRequest(
    modelOptions: ModelOptions,
    sessionMessage: SessionMessage,
  ): Promise<void> {
    if (!sessionMessage.content || typeof sessionMessage.content === 'string') return;

    const toolCalls = sessionMessage.content.filter(
      (part) => part.type === 'tool_call',
    ) as ToolCallPart[];

    const isParallel = modelOptions.parallelToolCalls ?? true;
    const toolNames = toolCalls.map((tc) => tc.name);
    this.logger.debug({ toolNames, isParallel }, 'Tool calls start');

    return await trace.startActiveSpan('execute_tool_calls', async (span) => {
      span.setAttribute('gen_ai.tool.names', toolNames.join(','));
      span.setAttribute('agent.tool.parallel', isParallel);

      try {
        if (isParallel) {
          const results = await Promise.all(
            toolCalls.map(async (toolCall) => {
              return await trace.startActiveSpan(
                `execute_tool ${toolCall.name}`,
                async (toolSpan) => {
                  toolSpan.setAttribute('gen_ai.operation.name', 'execute_tool');
                  toolSpan.setAttribute('gen_ai.tool.name', toolCall.name);
                  toolSpan.setAttribute('gen_ai.tool.call.id', toolCall.id);
                  try {
                    const result = await this.toolRegistry.execute(toolCall, {
                      signal: this.abortController.signal,
                    });
                    return result;
                  } catch (error) {
                    toolSpan.recordException(error as Error);
                    toolSpan.setStatus({
                      code: SpanStatusCode.ERROR,
                      message: (error as Error).message,
                    });
                    toolSpan.setAttribute('error.type', (error as Error).name || 'Error');
                    throw error;
                  } finally {
                    toolSpan.end();
                  }
                },
              );
            }),
          );
          const pendingToolResultMessage = this.sessionThread.addPendingMessage({
            type: 'pending_message',
            role: 'tool',
            name: sessionMessage.name,
            content: results,
          });
          this.logger.debug(
            {
              pendingMessage: {
                role: pendingToolResultMessage.role,
                pendingMessageId: pendingToolResultMessage.pendingMessageId,
                content: pendingToolResultMessage.content, // todo: use a function to summarize the content
                name: pendingToolResultMessage.name,
                refusal: pendingToolResultMessage.refusal,
              },
            },
            'Added to pending messages for tool results',
          );
        } else {
          const results = [] as ToolResultPart[];
          for (const toolCall of toolCalls) {
            // todo: implement toolcall options
            // todo: implement event callbacks
            const result = await trace.startActiveSpan(
              `execute_tool ${toolCall.name}`,
              async (toolSpan) => {
                toolSpan.setAttribute('gen_ai.operation.name', 'execute_tool');
                toolSpan.setAttribute('gen_ai.tool.name', toolCall.name);
                toolSpan.setAttribute('gen_ai.tool.call.id', toolCall.id);
                try {
                  const result = await this.toolRegistry.execute(toolCall, {
                    signal: this.abortController.signal,
                  });
                  return result;
                } catch (error) {
                  toolSpan.recordException(error as Error);
                  toolSpan.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: (error as Error).message,
                  });
                  toolSpan.setAttribute('error.type', (error as Error).name || 'Error');
                  throw error;
                } finally {
                  toolSpan.end();
                }
              },
            );
            results.push(result);
          }
          const pendingToolResultMessage = this.sessionThread.addPendingMessage({
            type: 'pending_message',
            role: 'tool',
            name: sessionMessage.name,
            content: results,
          });
          this.logger.debug(
            {
              pendingMessage: {
                role: pendingToolResultMessage.role,
                pendingMessageId: pendingToolResultMessage.pendingMessageId,
                content: pendingToolResultMessage.content, // todo: use a function to summarize the content
                name: pendingToolResultMessage.name,
                refusal: pendingToolResultMessage.refusal,
              },
            },
            'Added to pending messages for tool results',
          );
        }
      } catch (error) {
        span.recordException(error as Error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: (error as Error).message });
        span.setAttribute('error.type', (error as Error).name || 'Error');
        throw error;
      } finally {
        span.end();
      }
    });
  }

  /**
   * Get the history messages and if any of them was not in the same API mode as the current request,
   * convert it to the current transport's raw message format.
   * @param modelOptions
   * @returns
   */
  private getHistoryMessagesForTransport(modelOptions: ModelOptions): RawMessageType[] {
    const messages = [] as RawMessageType[];
    const transport = this.getTransport(modelOptions);
    for (const msg of this.sessionThread.threadMessages) {
      if (modelOptions.apiMode !== msg.apiMode || modelOptions.baseUrl !== msg.baseUrl) {
        if (this.convertedMessagesCache.has(msg.internalMessageId)) {
          messages.push(this.convertedMessagesCache.get(msg.internalMessageId)!);
          continue;
        }
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
        const sessionMsg = t.convertFromRawMessage(msg.rawMessage);
        const converted = transport.convertToRawMessage(sessionMsg);
        this.convertedMessagesCache.set(msg.internalMessageId, converted);
        messages.push(converted);
      } else {
        messages.push(msg.rawMessage);
      }
    }
    return messages;
  }

  private getTransport(options: ModelOptions): ProviderTransport<RawMessageType> {
    if (
      this.sessionThread.threadInfo.modelOptions.apiMode !== options.apiMode ||
      this.sessionThread.threadInfo.modelOptions.baseUrl !== options.baseUrl
    ) {
      const transport = PixiAgent.getTransport(options, this._transport);
      this.convertedMessagesCache.clear();
      this._transport = transport;
    }
    return this._transport;
  }

  private static peekPendingMessagesToExecute(thread: SessionThread): PendingMessage[] {
    const messages = [] as PendingMessage[];

    for (const msg of thread.getPendingMessages()) {
      if (
        messages.length == 0 ||
        (messages[messages.length - 1].role === msg.role &&
          messages[messages.length - 1].name === msg.name &&
          messages[messages.length - 1].refusal === msg.refusal)
      ) {
        messages.push(msg as PendingMessage);
      } else {
        break;
      }
    }
    return messages;
  }

  private static convertPendingMessages(
    thread: SessionThread,
    pendingMessages: PendingMessage[],
  ): SessionMessage {
    const content = [] as ContentPart[];
    for (const msg of pendingMessages) {
      if (typeof msg.content === 'string') {
        content.push({ type: 'text', text: msg.content });
      } else if (Array.isArray(msg.content)) {
        content.push(...msg.content);
      }
    }
    return {
      type: 'session_message',
      role: pendingMessages[0].role,
      content: content,
      name: pendingMessages[0].name,
      refusal: pendingMessages[0].refusal,
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

export type PixiAgentOptions = {
  /**
   * The timeout for the model request. If the model request takes longer than this time,
   * it will be aborted.
   *
   * The tool calls timeout is defined in the ToolRegistry.
   */
  modelRequestTimeout?: number;
  /**
   * The maximum number of iterations for the agent to execute.
   * If the agent executes more than this number, it will stop.
   */
  maxIterations?: number;
};
