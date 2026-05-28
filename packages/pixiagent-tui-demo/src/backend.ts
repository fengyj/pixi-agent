import type { 
  ContentPart,
  InternalMessage, 
  RawMessageType,
  SessionMessage, 
} from '@pixiagent/core';
import { 
  PixiAgent,
  UsageStats,  
  Session,
  Tools,
  Transports,
} from '@pixiagent/core';

type ModelOptions = Transports.ModelOptions;
const ToolRegistry = Tools.ToolRegistry;

export type ChatResponse = {
  text: string;
  usage: UsageStats;
  sessionUsage: UsageStats;
};

export type ChatBackend = {
  sendUserMessage(input: string): Promise<ChatResponse>;
  interrupt(reason?: string): void;
};

function extractTextFromSessionMessage(message: SessionMessage): string {
  const textChunks: string[] = [];

  const visitContent = (content: string | Array<ContentPart> | undefined): void => {
    if (!content) return;
    if (typeof content === 'string') {
      const trimmed = content.trim();
      if (trimmed.length > 0) {
        textChunks.push(trimmed);
      }
      return;
    }

    for (const part of content) {
      switch (part.type) {
        case 'text':
          if (part.text.trim().length > 0) {
            textChunks.push(part.text);
          }
          break;
        case 'thinking':
          if (part.content.trim().length > 0) {
            textChunks.push(part.content);
          }
          break;
        case 'refusal':
          if (part.reason.trim().length > 0) {
            textChunks.push(part.reason);
          }
          break;
        case 'tool_result':
          if (typeof part.result === 'string' && part.result.trim().length > 0) {
            textChunks.push(part.result);
          }
          break;
        default:
          break;
      }
    }
  };

  visitContent(message.content);

  const merged = textChunks.join(' ').trim();
  return merged.length > 0 ? merged : '(assistant returned no text content)';
}

function getSessionMessageFromRaw(
  rawMessage: RawMessageType | SessionMessage,
  apiMode: InternalMessage['apiMode'],
  baseUrl?: string,
): SessionMessage {
  if ((rawMessage as SessionMessage)?.type === 'session_message') {
    return rawMessage as SessionMessage;
  }

  const transport = Transports.getTransport(apiMode, baseUrl);
  return transport.convertFromRawMessage(rawMessage as RawMessageType);
}

function extractAssistantText(message: InternalMessage): string {
  const sessionMessage = getSessionMessageFromRaw(
    message.rawMessage,
    message.apiMode,
    message.baseUrl,
  );
  return extractTextFromSessionMessage(sessionMessage);
}

function findLastAssistantMessage(messages: InternalMessage[]): InternalMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant') {
      return messages[i];
    }
  }
  return undefined;
}

function normalizeUsage(usage: UsageStats | undefined): UsageStats {
  return {
    inputTokens: usage?.inputTokens ?? 0,
    outputTokens: usage?.outputTokens ?? 0,
    totalTokens: usage?.totalTokens ?? 0,
    cacheReadTokens: usage?.cacheReadTokens ?? 0,
    cacheCreatedTokens: usage?.cacheCreatedTokens ?? 0,
    reasoningTokens: usage?.reasoningTokens ?? 0,
    inputTokenDetails: usage?.inputTokenDetails ? { ...usage.inputTokenDetails } : undefined,
    outputTokenDetails: usage?.outputTokenDetails ? { ...usage.outputTokenDetails } : undefined,
  };
}

export class PixiAgentBackend implements ChatBackend {
  private readonly agent: PixiAgent;
  private readonly modelOptions: ModelOptions;

  constructor(
    modelOptions: ModelOptions,
    options: {
      modelRequestTimeout: number;
      maxIterations: number;
      maxModelRequestRetries: number;
    },
  ) {
    const session = Session.create({
      modelOptions,
    });
    const thread = Session.getDefaultThread(session);
    this.agent = new PixiAgent(thread, new ToolRegistry(), {
      modelRequestTimeout: options.modelRequestTimeout,
      maxIterations: options.maxIterations,
      maxModelRequestRetries: options.maxModelRequestRetries,
    });
    this.modelOptions = modelOptions;
  }

  async sendUserMessage(input: string): Promise<ChatResponse> {
    const userMessage: Omit<SessionMessage, 'messageId'> = {
      type: 'session_message',
      role: 'user',
      content: input,
    };

    const result = await this.agent.execute(this.modelOptions, userMessage);
    const requestUsage = normalizeUsage(result.usage);
    const sessionUsage = normalizeUsage(this.agent.sessionThread.session.totalUsage);

    const lastAssistant = findLastAssistantMessage(this.agent.sessionThread.threadMessages);
    const text = lastAssistant
      ? extractAssistantText(lastAssistant)
      : '(assistant did not return a message)';

    return {
      text,
      usage: requestUsage,
      sessionUsage,
    };
  }

  interrupt(reason?: string): void {
    this.agent.interrupt(reason);
  }
}
