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
import { Logger } from 'pino';
import { SessionThread } from './session';
import * as Transport from './transports';
import { ModelOptions } from './transports';
import { AgentExecutionLoop } from './agent-loop';
import { SessionMessage } from './message';
import { ToolRegistry } from './tools/tool';
import { PixiAgentErrorBuilder } from './errors';

import * as Observation from './observation';
import { AgentEventEmitter } from './event';
import { ContentParts } from './utils';
import { PixiAgentOptions, PixiAgentExecutionResult } from './agent';

export class AgentThreadRunner {
  private readonly logger: Logger;
  private readonly options: PixiAgentOptions;
  private readonly agentLoop: AgentExecutionLoop;
  private readonly pendingMessages: Array<
    Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' }
  > = [];
  private abortController = new AbortController();
  private isExecuting = false;

  constructor(
    public readonly sessionThread: SessionThread,
    private readonly toolRegistry: ToolRegistry,
    options?: PixiAgentOptions,
  ) {
    sessionThread.threadInfo.modelOptions = AgentThreadRunner.resolveApiModeAndBaseUrl(
      sessionThread.threadInfo.modelOptions,
    );
    this.logger = Observation.getLogger('agent').child({
      sessionId: sessionThread.session.sessionId,
      threadId: sessionThread.threadInfo.threadId,
    });
    this.options = options ?? {eventEmitter: new AgentEventEmitter()};
    this.agentLoop = new AgentExecutionLoop(
      this.sessionThread,
      this.toolRegistry,
      this.options,
      this.pendingMessages,
      this.logger,
    );
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
    modelOptions = AgentThreadRunner.resolveApiModeAndBaseUrl(modelOptions);

    if (this.isExecuting) {
      this.pendingMessages.push(input);
      return { action: 'steering' };
    }

    this.isExecuting = true;
    try {
      const result = await this.agentLoop.execute(modelOptions, input, this.abortController.signal);

      while (this.pendingMessages.length > 0) {
        const [msg, ...rest] = this.pendingMessages;

        if (rest.length > 0) {
          msg.content = rest.map((r) => r.content).reduce(ContentParts.concat, msg.content);
        }
        this.pendingMessages.length = 0;

        const nextResult = await this.agentLoop.execute(
          modelOptions,
          msg,
          this.abortController.signal,
        );

        if (result.action === 'execution' && nextResult.action === 'execution') {
          result.stopReason = nextResult.stopReason;
          result.messages.push(...nextResult.messages);
        }
      }

      return result;
    } finally {
      this.isExecuting = false;
      this.resetAbortControllerIfNeeded();
    }
  }

  public async interrupt(reason?: string): Promise<PixiAgentExecutionResult> {
    if (!this.isExecuting) {
      this.logger.debug('The agent is not running, no need to interrupt');
      return { action: 'abort', isTriggered: false };
    }
    if (this.abortController.signal.aborted) {
      this.logger.debug('The agent is already interrupted');
      return { action: 'abort', isTriggered: true };
    }
    reason = reason ?? 'User interrupted';
    this.logger.info({ reason }, 'Interrupting the agent execution');
    this.abortController.abort(PixiAgentErrorBuilder.agentInterrupted(reason));
    await this.options.eventEmitter.executionState
      .interrupted(this.sessionThread.session.sessionId, this.sessionThread.threadInfo.threadId)
      .catch((error) => {
        this.logger.error({ error }, 'An error occurred during interruption');
      });
    return { action: 'abort', isTriggered: true };
  }

  private resetAbortControllerIfNeeded(): void {
    if (this.abortController.signal.aborted) {
      this.abortController = new AbortController();
    }
  }

  private static resolveApiModeAndBaseUrl(modelOptions: ModelOptions): ModelOptions {
    const resolveResult = Transport.GlobalApiModeResolverRegistry.resolve(
      modelOptions.model,
      modelOptions.baseUrl,
      modelOptions.apiMode,
    );
    if (!resolveResult) {
      throw PixiAgentErrorBuilder.apiModeResolutionFailed(modelOptions.model, modelOptions.baseUrl);
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
}
