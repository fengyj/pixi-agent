import { PixiAgent } from '@pixiagent/core/agent';
import type { InternalMessage, SessionMessage, UsageStats } from '@pixiagent/core/message';
import { Session } from '@pixiagent/core/session';
import { ToolRegistry } from '@pixiagent/core/tool';
import type { ModelOptions } from '@pixiagent/core/transports';

export type ChatResponse = {
  text: string;
  usage: UsageStats;
  sessionUsage: UsageStats;
};

export type ChatBackend = {
  sendUserMessage(input: string): Promise<ChatResponse>;
};

function extractAssistantText(rawMessage: unknown): string {
  const textChunks: string[] = [];

  const visit = (value: unknown): void => {
    if (!value) return;
    if (typeof value === 'string') {
      if (value.trim().length > 0) {
        textChunks.push(value);
      }
      return;
    }

    if (Array.isArray(value)) {
      for (const item of value) visit(item);
      return;
    }

    if (typeof value === 'object') {
      const record = value as Record<string, unknown>;

      // Common text payloads across OpenAI/Anthropic-compatible responses.
      if (typeof record.text === 'string') {
        textChunks.push(record.text);
      }
      if (typeof record.content === 'string') {
        textChunks.push(record.content);
      }

      if (record.content && typeof record.content !== 'string') {
        visit(record.content);
      }
      if (record.output) {
        visit(record.output);
      }
      if (record.message) {
        visit(record.message);
      }
    }
  };

  visit(rawMessage);

  const merged = textChunks.join('').trim();
  return merged.length > 0 ? merged : '(assistant returned no text content)';
}

function findLastAssistantMessage(messages: InternalMessage[]): InternalMessage | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === 'assistant') {
      return messages[i];
    }
  }
  return undefined;
}

function createEmptyUsage(): UsageStats {
  return {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cacheReadTokens: 0,
    cacheCreatedTokens: 0,
    reasoningTokens: 0,
  };
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
    const userMessage: SessionMessage = {
      type: 'session_message',
      role: 'user',
      content: input,
    };

    const result = await this.agent.execute(this.modelOptions, userMessage);
    const requestUsage = normalizeUsage(result.usage);
    const sessionUsage = normalizeUsage(this.agent.sessionThread.session.totalUsage);

    const lastAssistant = findLastAssistantMessage(this.agent.sessionThread.threadMessages);
    const text = lastAssistant ? extractAssistantText(lastAssistant.rawMessage) : '(assistant did not return a message)';

    return {
      text,
      usage: requestUsage,
      sessionUsage,
    };
  }
}
