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
import { Session, SessionThread } from './session';
import * as Transport from './transports';
import { ModelOptions } from './transports';
import { SessionMessage } from './message';
import { ToolRegistry } from './tools/tool';
import { PixiAgentErrorBuilder } from './errors';

import * as Observation from './observation';
import { AgentEventEmitter } from './event';
import { ContentParts } from './utils';
import { AgentThreadRunner } from './agent-thread-runner';

export class PixiAgent {
  private readonly logger: Logger;
  private readonly threadRunners: Map<string, AgentThreadRunner> = new Map();
  private readonly options: PixiAgentOptions;

  constructor(
    public readonly session: Session,
    private readonly toolRegistry: ToolRegistry,
    options?: PixiAgentOptions,
  ) {
    this.logger = Observation.getLogger('agent').child({
      sessionId: session.sessionId,
    });
    this.options = options ?? { eventEmitter: new AgentEventEmitter() };
  }

  /**
   * The execute function is the main loop of the agent.
   * It takes the input, and execute the tasks in the input.
   * The input is a pending message, which is added to the session thread's pending messages.
   * The agent will peek the pending messages, and execute them one by one.
   * @param modelOptions
   * @param input
   * @param threadId
   */
  public async execute(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' },
    threadId?: string,
  ): Promise<PixiAgentExecutionResult> {
    const threadRunner = this.getThreadRunner(threadId);
    return await threadRunner.execute(modelOptions, input);
  }

  public async interrupt(reason?: string, threadId?: string): Promise<PixiAgentExecutionResult> {
    const threadRunner = this.getThreadRunner(threadId);
    return await threadRunner.interrupt(reason);
  }

  public async fork(
    modelOptions: ModelOptions,
    input: Omit<SessionMessage, 'messageId' | 'role'> & { messageId?: string; role: 'user' },
    fromthreadId: string,
    fromMessageId: string,
    isAnnotation: boolean = false,
  ): Promise<PixiAgentExecutionResult> {}

  private getThreadRunner(threadId?: string): AgentThreadRunner {
    if (!threadId) {
      threadId = this.session.defaultThread;
    }
    if (this.threadRunners.has(threadId)) {
      return this.threadRunners.get(threadId)!;
    }
    const sessionThread = Session.getThreads(this.session, threadId);
    if (!sessionThread) {
      throw PixiAgentErrorBuilder.threadNotFound(threadId);
    }
    const threadRunner = new AgentThreadRunner(
      sessionThread as SessionThread,
      this.toolRegistry,
      this.options,
    );
    this.threadRunners.set(threadId, threadRunner);
    return threadRunner;
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
  eventEmitter: AgentEventEmitter;
}

export type AgentStopReason =
  | 'end_turn'
  | 'max_turn_requests'
  | 'max_tokens'
  | 'refusal'
  | 'cancelled';

export type PixiAgentExecutionResult =
  | {
      action: 'execution';
      stopReason: AgentStopReason;
      messages: SessionMessage[];
    }
  | {
      action: 'abort';
      isTriggered: boolean;
    }
  | {
      action: 'compact';
    }
  | {
      action: 'steering';
    };
