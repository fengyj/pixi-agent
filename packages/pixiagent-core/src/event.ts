import { type AgentStopReason } from './agent';
import {
  ContentPart,
  ModelStopReasons,
  RoleType,
  SessionMessage,
  ToolCallPart,
  ToolResultPart,
  UsageStats,
} from './message';
import Emittery, { type EventDataPair, type UnsubscribeFunction } from 'emittery';
import { nanoid } from 'nanoid';
import { StreamCallbacks } from './transports';

export type AgentEventData = AgentMessageChunkEventData;

interface AgentMessageChunkEventData {
  eventType: 'message_chunk';
  role: RoleType;
  sessionId: string;
  threadId: string;
  /**
   * When messageId is not undefined, it means all the contentparts of the message have been sent.
   * For the ongoing process, the chunks usually are sent by the transport, and the final chuck
   * with the messageId is sent by the PixiAgent. The chunk with an empty TextPart.
   * But for the replay, the messageId will be attached to the last content part of the session message.
   */
  messageId?: string;
  previousMessageId: string | null;
  /**
   * When assistant message is done, usage will be attached when it's available.
   */
  usage?: UsageStats;
  /**
   * When assistant message is done, the stop reason will be attached when it's available.
   */
  stopReason?: ModelStopReasons;
  /**
   * Only available when `after_model_response` event is emitted and the stop reason is `refusal`.
   */
  refusal?: string;
  /**
   * An id that identifies the chuncks belong to the same message.
   *
   * Basically, it can be treated as a temporary message id. The real message id will be set
   * when the message is complete, and the `isFinal` flag is set to true. That because
   * the message id is generated only when it is added to the session thread.
   */
  id: string;
  /**
   * The index of the contentpart in the message. The chunks belonging to the same contentPartIndex should
   * be merged together as a whole ContentPart data.
   */
  contentPartIndex: number;
  /**
   * The value of the chunk may not be a complete ContentPart, need to be merged with
   * other chunks with the same contentPartIndex to form a complete ContentPart.
   */
  chunk: ContentPart;
  /**
   * The index of the partial chunk of the ContentPart.
   */
  chunkIndex: number;
}

interface AgentExecutionStateChangedEventData {
  eventType: 'execution_state_changed';
  newState: /**
     * User message has been received, but has not yet been added to the session thread.

     * Happend before the `PixiAgent.execute` method is called.
     * For the scenarios like verify the user input, and determine whether to continue 
     * the execution or not, the user message can be intercepted and rejected 
     * before it is added to the session thread.
     */
    | 'user_message_received'
    /**
     * The user message has been added to the session thread,
     * but the model request has not yet been sent.
     */
    | 'before_model_request'
    /**
     * The model has responded, and has been added to the session thread.
     */
    | 'after_model_response'
    /**
     * Before invoking the tool.
     *
     * This event is emitted before a tool is called, allowing for any necessary preparations
     * or checks.
     */
    | 'before_tool_call_request'
    /**
     * After the tool has responded.
     */
    | 'after_tool_call_response'
    /**
     * All the tool calls have finished, and the tool message has been
     * added to the session thread.
     */
    | 'tool_calls_finished'
    /**
     * Once all the tool calls have been completed, or the llm response without tool calls,
     * this event is emitted.
     */
    | 'iteration_completed'
    /**
     * Emitted when the interrupt signal is triggered. The agent may haven't been stopped yet.
     */
    | 'interrupted'
    /**
     * The execution has been completed.
     */
    | 'completed'
    /**
     * Any error happened during the execution, abort or timeout, or any other unexpected error.
     * The agent finished abnormally.
     */
    | 'incomplete';
  /**
   * Avaiable when `user_message_received`, `before_model_request`, `after_model_response`,
   * and `after_tool_call_response` events are emitted.
   */
  message?: Omit<SessionMessage, 'messageId'> | SessionMessage;
  /**
   * Available when `after_model_response`.
   */
  hasToolCall?: boolean;
  /**
   * Available when `after_model_response`. And when 'completed' or 'incomplete',
   * the usage is the total usage of the whole execution.
   */
  usage?: UsageStats;
  /** Only available when `after_model_response` event is emitted and the stop reason is `refusal`. */
  refusal?: string;
  /** Only available when `after_model_response` event is emitted. */
  modelStopReason?: ModelStopReasons;
  /**
   * Available when `after_model_response`, `after_tool_call_response`,
   * and `iteration_completed` events are emitted.
   * Starts from 0.
   */
  iteration?: number;
  /**
   * Available when `before_tool_call_request` event is emitted.
   */
  toolCall?: ToolCallPart;
  /**
   * Available when `before_tool_call_request` and `after_tool_call_response` events are emitted.
   */
  toolResult?: ToolResultPart;
  /**
   * Available when the execution is interrupted or completed.
   */
  agentStopReason?: AgentStopReason;
  /**
   * The error that caused the execution to stop abnormally.
   * It may be available when `error` event is emitted.
   */
  error?: Error;
}

export type AgentEvents = {
  message_chunk: AgentMessageChunkEventData;
  execution_state_changed: AgentExecutionStateChangedEventData;
};

export class AgentEventEmitter {
  private emitter = new Emittery<AgentEvents>();
  public executionState = {
    userMessageReceived: (msg: Omit<SessionMessage, 'messageId'>): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'user_message_received',
        message: msg,
      });
    },
    beforeModelRequest: (msg: SessionMessage): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'before_model_request',
        message: msg,
      });
    },
    afterModelResponse: (msg: SessionMessage): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'after_model_response',
        message: msg,
        hasToolCall:
          Array.isArray(msg.content) && msg.content.some((part) => part.type === 'tool_call'),
        usage: msg.modelResponseInfo?.usage,
        modelStopReason: msg.modelResponseInfo?.stopReason,
        refusal:
          msg.modelResponseInfo?.stopReason === 'refusal'
            ? msg.modelResponseInfo.refusal
            : undefined,
      });
    },
    beforeToolCallRequest: (toolCall: ToolCallPart): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'before_tool_call_request',
        toolCall,
      });
    },
    afterToolCallResponse: (toolCall: ToolCallPart, toolResult: ToolResultPart): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'after_tool_call_response',
        toolCall,
        toolResult,
      });
    },
    toolCallsFinished: (msg: SessionMessage): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'tool_calls_finished',
        message: msg,
      });
    },
    iterationCompleted: (iteration: number): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'iteration_completed',
        iteration,
      });
    },
    executionFinished: (
      agentStopReason: AgentStopReason,
      usage?: UsageStats,
      error?: Error,
    ): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: agentStopReason === 'end_turn' ? 'completed' : 'incomplete',
        agentStopReason: agentStopReason,
        usage,
        error,
      });
    },
    interrupted: (): Promise<void> => {
      return this.emit('execution_state_changed', {
        eventType: 'execution_state_changed',
        newState: 'interrupted',
      });
    },
  };
  messageChunk = {
    chunk: (
      sessionId: string,
      threadId: string,
      role: RoleType,
      id: string,
      previousMessageId: string | null,
      contentPartIndex: number,
      chunk: ContentPart,
      chunkIndex: number,
    ): Promise<void> => {
      return this.emit('message_chunk', {
        eventType: 'message_chunk',
        sessionId,
        threadId,
        role,
        id,
        previousMessageId,
        contentPartIndex,
        chunk,
        chunkIndex,
      });
    },
    finish: (
      sessionId: string,
      threadId: string,
      role: RoleType,
      id: string,
      previousMessageId: string | null,
      messageId: string,
      contentPartIndex: number,
      usage?: UsageStats,
      stopReason?: ModelStopReasons,
      refusal?: string,
    ): Promise<void> => {
      return this.emit('message_chunk', {
        eventType: 'message_chunk',
        sessionId,
        threadId,
        role,
        id,
        previousMessageId,
        messageId,
        usage,
        stopReason,
        refusal,
        contentPartIndex,
        chunk: { type: 'text', text: '' },
        chunkIndex: 0,
      });
    },
  };

  on(
    eventName: keyof AgentEvents,
    listener: (event: EventDataPair<AgentEvents, typeof eventName>) => void | Promise<void>,
    options?: { signal?: AbortSignal },
  ): UnsubscribeFunction {
    return this.emitter.on(eventName, listener, options);
  }

  off(
    eventName: keyof AgentEvents,
    listener: (event: EventDataPair<AgentEvents, typeof eventName>) => void | Promise<void>,
  ): void {
    this.emitter.off(eventName, listener);
  }

  async emit(
    name: keyof AgentEvents,
    data: AgentEvents[typeof name],
    isSerial: boolean = false,
  ): Promise<void> {
    if (isSerial) {
      await this.emitter.emitSerial(name, data);
    } else {
      await this.emitter.emit(name, data);
    }
  }
}

export const AgentMessageChunkEventCallbacks = {
  create: (
    eventEmitter: AgentEventEmitter,
    sessionId: string,
    threadId: string,
    role: RoleType,
    previousMessageId: string | null,
  ): StreamCallbacks => {
    const id = nanoid(10);
    let maxContentPartIndex: number = -1;
    return {
      onChunk: (chunk): Promise<void> => {
        maxContentPartIndex = Math.max(maxContentPartIndex, chunk.contentPartIndex);
        return eventEmitter.messageChunk.chunk(
          sessionId,
          threadId,
          role,
          id,
          previousMessageId,
          chunk.contentPartIndex,
          chunk.contentPartChunk,
          chunk.chunkIndex,
        );
      },
      onFinish: (msg): Promise<void> => {
        return eventEmitter.messageChunk.finish(
          sessionId,
          threadId,
          role,
          id,
          previousMessageId,
          msg.messageId,
          maxContentPartIndex + 1,
          msg.modelResponseInfo?.usage,
          msg.modelResponseInfo?.stopReason,
          msg.modelResponseInfo?.stopReason === 'refusal'
            ? msg.modelResponseInfo?.refusal
            : undefined,
        );
      },
    };
  },
};
