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
import { ContentPart, RawMessageType, SessionMessage, ToolCallPart, ToolResultPart } from './message';
import { ToolRegistry } from './tool';

export class PixiAgent {
  private _transport: ProviderTransport<RawMessageType>;
  private convertedMessagesCache = new Map<string, RawMessageType>();
  private transportCache = new Map<string, ProviderTransport<RawMessageType>>();
  // todo: add event listener as parameter to expose the events.
  constructor(
    public sessionThread: SessionThread,
    public toolRegistry: ToolRegistry,
  ) {
    sessionThread.threadInfo.modelOptions = PixiAgent.resolveApiModeAndBaseUrl(
      sessionThread.threadInfo.modelOptions,
    );
    this._transport = PixiAgent.getTransport(sessionThread.threadInfo.modelOptions);
  }

  /**
   * The execute function is the main loop of the agent.
   * It takes the input, and execute the tasks in the input.
   * The input is a pending message, which is added to the session thread's pending messages.
   * The agent will peek the pending messages, and execute them one by one.
   * @param modelOptions
   * @param input todo: add InterruptionMessage
   */
  public async execute(modelOptions: ModelOptions, input: PendingMessage): Promise<void> {
    modelOptions = PixiAgent.resolveApiModeAndBaseUrl(modelOptions);
    // todo: check the input if it's an interruption, if so,
    // handle the interruption (clear the pending messages, and interrupt the unfinished execution).)

    this.sessionThread.addPendingMessage(input);

    while (this.sessionThread.getPendingMessages().length > 0) {
      const pendingMessages = PixiAgent.peekPendingMessagesToExecute(this.sessionThread);
      const sessionMessage = PixiAgent.convertPendingMessages(this.sessionThread, pendingMessages);

      if (
        !sessionMessage.refusal &&
        (!sessionMessage.content ||
          (typeof sessionMessage.content !== 'string' && sessionMessage.content.length === 0))
      ) {
        this.sessionThread.removePendingMessage(pendingMessages.map((msg) => msg.pendingMessageId));
        // todo: add a warning log here for the invalid pending message.
        continue;
      }

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
      // todo: a hook here can be used for session persistence.
    }
  }

  public async interrupt(): Promise<void> {
    // todo: implement this.
  }

  private async executeLLMRequest(
    modelOptions: ModelOptions,
    sessionMessage: SessionMessage,
  ): Promise<void> {
    const transport = this.getTransport(modelOptions);
    const historyMessages = this.getHistoryMessagesForTransport(modelOptions);
    const rawMessage = transport.convertToRawMessage(sessionMessage);
    // todo: implement event callbacks
    // todo: implement llm request options
    const response = await transport.generate(
      modelOptions,
      [...historyMessages, rawMessage],
      {},
      {},
    );
    this.sessionThread.addMessage(rawMessage, modelOptions);
    const respSessionMsg = transport.convertFromRawMessage(response.rawMessage);
    if (
      !respSessionMsg.refusal &&
      respSessionMsg.content &&
      typeof respSessionMsg.content !== 'string'
    ) {
      const parts = respSessionMsg.content! as ContentPart[];
      const toolParts = parts.filter((part) => part.type === 'tool_call') as ContentPart[];
      if (toolParts.length > 0) {
        this.sessionThread.addPendingMessage({
          type: 'pending_message',
          role: 'assistant',
          name: sessionMessage.name,
          content: toolParts,
        } as PendingMessage);
      }
    }
    this.sessionThread.addMessage(response.rawMessage, modelOptions, response.usage);
  }

  private async executeToolCallRequest(
    modelOptions: ModelOptions,
    sessionMessage: SessionMessage,
  ): Promise<void> {
    if (!sessionMessage.content || typeof sessionMessage.content === 'string') return;

    const toolCalls = sessionMessage.content.filter(
      (part) => part.type === 'tool_call',
    ) as ToolCallPart[];
    if (modelOptions.parallelToolCalls ?? true) {
      const results = await Promise.all(
        toolCalls.map(async (toolCall) => {
          const result = await this.toolRegistry.execute(toolCall, {});
          return result;
        }),
      );
      this.sessionThread.addPendingMessage({
        type: 'pending_message',
        role: 'tool',
        name: sessionMessage.name,
        content: results,
      } as PendingMessage);
    } else {
      const results = [] as ToolResultPart[];
      for (const toolCall of toolCalls) {
        // todo: implement toolcall options
        // todo: implement event callbacks
        const result = await this.toolRegistry.execute(toolCall, {});
        results.push(result);
      }
      this.sessionThread.addPendingMessage({
        type: 'pending_message',
        role: 'tool',
        name: sessionMessage.name,
        content: results,
      } as PendingMessage);
    }
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
      throw new Error(
        `Cannot resolve the API mode for model ${modelOptions.model} with baseUrl ${modelOptions.baseUrl}`,
      );
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
      undefined,
      dialectResolver,
    );
  }
}
