import { type AgentStopReason } from './agent';
import { ContentPart, RoleType, SessionMessage, ToolCallPart, ToolResultPart, UsageStats } from './message';
import mitt from 'mitt';

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
  previousMessageId?: string;
  /**
   * When assistant message is done, usage will be attached when it's available.
   */
  usage?: UsageStats;
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
  /**
   * Indicates whether is the first partial chunk of the ContentPart.
   */
  isChunkBeginning: boolean;
  /**
   * Indicates whether is the last partial chunk of the ContentPart.
   */
  isChunkFinal: boolean;
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
  stopReason?: AgentStopReason;
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
  private emitter = mitt<AgentEvents>();

  public on<K extends keyof AgentEvents>(eventType: K, handler: (data: AgentEvents[K]) => void) {
    this.emitter.on(eventType, handler);
  }

  public off<K extends keyof AgentEvents>(eventType: K, handler: (data: AgentEvents[K]) => void) {
    this.emitter.off(eventType, handler);
  }

  public emit<K extends keyof AgentEvents>(data: AgentEvents[K]) {
    this.emitter.emit(data.eventType, data);
  }
}
