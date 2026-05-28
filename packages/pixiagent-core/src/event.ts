import { ContentPart, SessionMessage, ToolResultPart, UsageStats } from "./message";



export type AgentEventType = AgentMessageEventType | AgentToolEventType | AgentSessionEventType;



type AgentMessageEventType = 'message';
type AgentMessageChunkEventType = 'message_chunk' | 'thinking_chunk' | 'tool_call_chunk' | 'refusal_chunk';
type AgentToolEventType = 'tool_call' | 'tool_call_update';
type AgentSessionEventType = 'session_updated';


interface AgentMessageEventData {
    /**
     * InternalMessageId
     */
    messageId: string;
    message: SessionMessage;
    sessionId: string;
    threadId: string;
    usage?: UsageStats;
    previousMessageId?: string;
    createdAt?: string;
    /**
     * This can be used to map the chuncks to the message.
     */
    chunckGroupId?: string;
}

interface AgentMessageChunkEventData {
    /**
     * An id that identifies the chuncks belong to the same message.
     */
    chunckGroupId: string;
    role: 'user' | 'assistant' | 'tool';
    chunk: string | Exclude<ContentPart, ToolResultPart>;
    sessionId: string;
    threadId: string;
    isBeginning: boolean;
    isFinal: boolean;
    previousMessageId?: string;
}