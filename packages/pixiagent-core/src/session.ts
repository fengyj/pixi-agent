import { InternalMessage, RawMessageType, RoleType, SessionMessage, UsageStats } from './message';
import { ModelProvider } from './model';
import { ModelOptions } from './transports';
import { nanoid } from 'nanoid';

/**
 * Session ID: nano(8)
 * Thread ID: Session ID + seq number (2 digits, basic range: 01~99, extend range: A0~ZZ)
 * Message ID: Thread ID + seq number (2 digits, basic range: 01~99, extend range: A0~ZZ)
 */

/**
 * The session is a context of the agent activities.
 *
 * It includes the messages, and some other information. A session is binded to a host,
 * which means the session is only valid for the specific host. If want to continue
 * the conversation in another host, has to handover it to the new one. The agent can
 * handover the session voluntarily or asked fron another host. For example, user wants to
 * continue the work from local to cloud, or takeover the session to mobile device. By doing so,
 * we don't have to worry about the concurrency issue.
 *
 * And the conversation is not linear. The user can fork from a message to discuss anther topic,
 * or the agent can delegate a subtask to another agent. And the compression of the messages also
 * can be thought as a kind of fork. The compressed message is forked from an assistant message,
 * and joined back to the last (n) message(s) in the original conversation.
 *
 */
export interface Session {
  sessionId: string;
  title: string | null;
  createdAt: string;
  updatedAt: string;
  /**
   * Marks the session's holder.
   */
  holder: string | null;
  /**
   * The parent session id if this session is forked from another session.
   * It is used for tracking the session lineage, and can be used for UI to display the session tree structure.
   */
  parentSessionId: string | null;
  /**
   * The messages no matter in which threads.
   */
  messages: InternalMessage[];
  /**
   * This is an idea to simulate git's branch concept. Conversations need not to be linear, it can be
   * emanative. User can have different discussions on a particular message or pose a sidecar question.
   * For example, a user may simply want the AI to explain a word they don't know
   * and prefer not to discuss it in the main thread.
   */
  threads: SessionThreadInfo[];
  /**
   * This is for the UI to know which thread is used for the next-turn conversation by default.
   *
   * It should be set by the user or the agent when the session is created.
   * For example, when the user want to start a new discussion on a particular message,
   * they can set the default_thread to the new thread head,
   * so that the UI can automatically switch to the new thread for the next-turn conversation.
   */
  defaultThread: string;
  /**
   * Session level total usage
   */
  totalUsage: UsageStats;
  /**
   * Can be used to store any extra information.
   * For example, if want to use a database to store the session data,
   * we may save the last message has been saved, so just nedd save the new messages next time.
   */
  metadata?: Record<string, string>;
  /**
   * The media sources used in the session.
   *
   * For the media which are not attached as base64 in the message.
   * If the media is attached as URL, the URL could be a temporary presigned URL
   * with an expiration time, so the info can be used to refresh the URL when needed.
   * And if the media is uploaded to a provider, it can be expired, or user can use another provider
   * to continue the conversation, so we can replace the mediaSource in the message.
   */
  mediaSources?: Record<string, MediaSourceInfo>;
}

export interface MediaSourceInfo {
  mediaSourceId: string;
  /** the location can be a file path, or a URL. Which can be used for the agent to get the file */
  location: string;
  fileName?: string;
  mimeType?: string;
  size?: number;
  /** The url can be used by LLM providers to access the media */
  url?: string;
  urlExpireAt?: number;
  /** Records of the uploads. Key is the provider */
  uploaded?: Record<
    string,
    {
      /** The file ID returned by the provider */
      fileId: string;
      /** The name of the provider. Like 'openai' */
      provider: ModelProvider;
      /** The expiration time of the uploaded file */
      expireAt?: number;
    }
  >;
}

export interface SessionThreadInfo {
  /**
   * Thread id, it's a constant value.
   */
  threadId: string;
  /**
   * The title of the thread.
   */
  title: string | null;
  /**
   * There are two kinds of thread, one is branch style, the other is annotation style.
   * - Branch: the thread is forked from a particular message, and the messages before that
   *           will be included in the thread. This is the most common style, and in this style,
   *           the rootMessageId is undefined.
   * - Annotation: the thread is forked from a particular message, but the messages before that
   *               will not be included in the thread. This style is usually used for the sidecar question,
   *               and in this style, the rootMessageId is the message id of the particular message.
   */
  rootMessageId: string | null;
  /**
   * A mark to indicate that the thread is forked from a particular message. 
   * It can be used to track the thread lineage. And when user wants to delete a message,
   * we can know if there are any threads should be marked as deleted.
   */
  forkedFromMessageId: string | null;
  /**
   * The head message id of the thread.
   * When there is no messages in the session, it's undefined.
   */
  headMessageId?: string;
  /**
   * the model options used in previously. So that when user changed model or some others,
   * we may know if needs to use a new transports or a new dialect.
   */
  modelOptions: ModelOptions;
  /**
   * The thread level usage
   */
  totalUsage: UsageStats;
  /** Indicates how many LLM invocations have occurred in this thread */
  totalTurns: number;
  createdAt: string;
  updatedAt: string;
}

/**
 * The session thread is a linear conversation flow. It is used for the conversation with the LLM.
 *
 * It's a transit data (Session data is used for persistence, and SessionThread is used for the runtime)
 */
export class SessionThread {
  constructor(
    public readonly session: Session,
    public readonly threadInfo: SessionThreadInfo,
    public readonly threadMessages: InternalMessage[],
  ) {}

  addMessage(
    role: RoleType,
    rawMessage:
      | Omit<RawMessageType, 'messageId'>
      | (Omit<SessionMessage, 'messageId'> & { messageId?: string }),
    modelOptions: ModelOptions,
    usage?: UsageStats,
    createdAt?: string,
  ): InternalMessage {
    if (
      role === 'assistant' &&
      (rawMessage as Omit<SessionMessage, 'messageId'>)?.type === 'session_message'
    ) {
      throw new Error('Assistant message cannot be a SessionMessage');
    } else if (
      role !== 'assistant' &&
      (rawMessage as Omit<SessionMessage, 'messageId'>)?.type !== 'session_message'
    ) {
      throw new Error('User or tool message must be a SessionMessage');
    }
    const newMessageId = getSeqId(this.threadInfo.threadId, this.threadMessages.length + 1);
    const rawMsg = {
      ...rawMessage,
      messageId: 'messageId' in rawMessage ? rawMessage.messageId || newMessageId : newMessageId,
    } as RawMessageType | SessionMessage;

    const lastMessageId =
      this.threadMessages.length > 0
        ? this.threadMessages[this.threadMessages.length - 1].internalMessageId
        : undefined;
    const now = new Date().toISOString();
    createdAt = createdAt || now;
    const newMessage: InternalMessage = {
      internalMessageId: newMessageId,
      model: modelOptions.model,
      apiMode: modelOptions.apiMode!,
      baseUrl: modelOptions.baseUrl,
      rawMessage: rawMsg,
      role: role,
      previousMessageId: lastMessageId,
      createdAt: createdAt,
      completedAt: now,
      usage: usage,
    };
    this.threadMessages.push(newMessage);
    this.session.messages.push(newMessage);
    this.session.updatedAt = now;
    this.threadInfo.headMessageId = newMessage.internalMessageId;
    this.threadInfo.updatedAt = now;
    if (newMessage.role === 'assistant') {
      this.threadInfo.totalTurns += 1;
      this.threadInfo.totalUsage = UsageStats.sum(this.threadInfo.totalUsage, usage);
      this.session.totalUsage = UsageStats.sum(this.session.totalUsage, usage);
    }
    return newMessage;
  }
}

export const Session = {
  /**
   * create a new session.
   */
  create: createSession,
  /**
   * Get threads from a session.
   */
  getThreads: getThreadsFromSession,
  /**
   * Get the default thread from a session.
   * @param session
   * @returns
   */
  getDefaultThread(session: Session): SessionThread {
    return this.getThreads(session, session.defaultThread) as SessionThread;
  },
  /**
   * Fork a new thread from a particular message in the session.
   */
  fork: forkThread,
};

function createSessionId(length: number = 8): string {
  return nanoid(length);
}

function getSeqId(id: string, num: number, digits = 2): string {
  if (!Number.isInteger(num) || num < 0) {
    throw new RangeError('num must be a non-negative integer');
  }
  if (!Number.isInteger(digits) || digits <= 0) {
    throw new RangeError('digits must be a positive integer');
  }

  const threshold = 10 ** digits;
  if (num < threshold) {
    return `${id}${String(num).padStart(digits, '0')}`;
  }

  // Extended range uses a leading letter for overflow beyond the decimal digits,
  // while the remaining positions are encoded in base-36.
  //
  // Examples:
  //   digits=2: 99 -> ABC99, 100 -> ABCA0, 135 -> ABCAZ, 136 -> ABCB0
  //   digits=3: 99 -> ABC099, 999 -> ABC999, 1000 -> ABCA00, 1010 -> ABCA0A
  const offset = num - threshold;
  const suffixLength = Math.max(0, digits - 1);
  const blockSize = 36 ** suffixLength;
  const maxOffset = 26 * blockSize - 1;

  if (offset > maxOffset) {
    throw new RangeError(`num is too large to encode with ${digits} digits`);
  }

  const prefixIndex = Math.floor(offset / blockSize);
  const prefix = String.fromCharCode(65 + prefixIndex);

  let remainder = offset % blockSize;
  let suffix = '';

  for (let i = 0; i < suffixLength; i += 1) {
    const digit = remainder % 36;
    suffix = String.fromCharCode(digit < 10 ? 48 + digit : 55 + digit) + suffix;
    remainder = Math.floor(remainder / 36);
  }

  if (suffix.length < suffixLength) {
    suffix = suffix.padStart(suffixLength, '0');
  }

  return `${id}${prefix}${suffix}`;
}

/**
 * create a new session.
 * @param options
 * @param message the first message of the session
 * @returns
 */
function createSession(options: {
  modelOptions: ModelOptions;
  title?: string;
  holder?: string;
  parentSessionId?: string;
}): Session {
  const sessionId = createSessionId();
  const now = new Date().toISOString();
  const session: Session = {
    sessionId,
    createdAt: now,
    updatedAt: now,
    holder: options.holder ?? null,
    parentSessionId: options.parentSessionId ?? null,
    title: options.title ?? null,
    messages: [],
    threads: [],
    defaultThread: getSeqId(sessionId, 1),
    totalUsage: UsageStats.empty(),
  };
  forkThread(session, false, options.modelOptions);
  return session;
}

/**
 *
 * @param session
 * @param threadId
 * @returns
 */
function getThreadsFromSession(
  session: Session,
  threadId?: string,
): SessionThread | SessionThread[] | undefined {
  const messages = session.messages.reduce(
    (acc, message) => {
      acc[message.internalMessageId] = message;
      return acc;
    },
    {} as Record<string, InternalMessage>,
  );

  const threads = threadId ? undefined : ([] as SessionThread[]);
  for (const thread of session.threads) {
    if (threadId && thread.threadId !== threadId) continue;

    const threadMessages: InternalMessage[] = [];
    let currentMessageId = thread.headMessageId;
    while (currentMessageId) {
      const message = messages[currentMessageId];
      if (!message) {
        throw new Error(`Message with id ${currentMessageId} not found in session messages`);
      }
      threadMessages.unshift(message);
      currentMessageId =
        thread.rootMessageId && thread.rootMessageId === message.internalMessageId
          ? undefined
          : message.previousMessageId;
    }

    const sessionThread = new SessionThread(session, thread, threadMessages);
    if (threadId) {
      return sessionThread;
    } else {
      threads!.push(sessionThread);
    }
  }
  return threadId ? undefined : threads;
}

/**
 * Fork a new thread from a particular message in the session.
 * @param session
 * @param fromMessageId
 * @param includeHistory
 * @param pendingMessage
 * @returns
 */
function forkThread(
  session: Session,
  includeHistory: boolean,
  modelOptions: ModelOptions,
  fromMessageId?: string,
): SessionThread {
  const threadId = getSeqId(session.sessionId, session.threads.length + 1);
  const threadInfo: SessionThreadInfo = {
    threadId: threadId,
    rootMessageId: includeHistory ? null : (fromMessageId ?? null),
    forkedFromMessageId: includeHistory ? null : (fromMessageId ?? null),
    headMessageId: fromMessageId,
    modelOptions: modelOptions,
    title: null,
    totalUsage: UsageStats.empty(),
    totalTurns: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  session.threads.push(threadInfo);
  const thread = getThreadsFromSession(session, threadInfo.threadId) as SessionThread;
  return thread;
}
