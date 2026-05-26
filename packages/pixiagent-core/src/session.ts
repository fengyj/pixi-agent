
import {
  InternalMessage,
  RawMessageType,
  UsageStats,
} from './message';
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
  title?: string;
  createdAt: string;
  updatedAt: string;
  /**
   * Marks the session's holder.
   */
  holder?: string;
  /**
   * The parent session id if this session is forked from another session.
   * It is used for tracking the session lineage, and can be used for UI to display the session tree structure.
   */
  parentSessionId?: string;
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
   * This is used for recording the sources of the media (image, video, audio, document, etc.)
   */
  mediaInfo?: MediaInfo[];
}

/**
 * This is used for recording the sources of the media (image, video, audio, document, etc.) 
 * information in the messages when they were attached as files or URLs. This data can be
 * used when the URL or the file id is expired or invalid, so that the media can be re-uploaded 
 * or re-fetched.
 */
export interface MediaInfo {
  /**
   * The value can be used to retrieve the media from the original source. It can be a URL,
   * or a file path, or something else, like a key of the record in the database.
   */
  originalKey: string;
  /**
   * The value used by the message. Can be a URL or file id.
   */
  key: string;
  /**
   * If the key has a expiration time, it can be used to determine if need to create a new 
   * key or not.
   */
  expireAt?: number;
}

export interface SessionThreadInfo {
  /**
   * Thread id, it's a constant value.
   */
  threadId: string;
  /**
   * The title of the thread.
   */
  title?: string;
  /**
   * There are two kinds of thread, one is branch style, the other is annotation style.
   * - Branch: the thread is forked from a particular message, and the messages before that
   *           will be included in the thread. This is the most common style, and in this style,
   *           the rootMessageId is undefined.
   * - Annotation: the thread is forked from a particular message, but the messages before that
   *               will not be included in the thread. This style is usually used for the sidecar question,
   *               and in this style, the rootMessageId is the message id of the particular message.
   */
  rootMessageId?: string;
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
    public session: Session,
    public threadInfo: SessionThreadInfo,
    public threadMessages: InternalMessage[],
  ) {}

  addMessage(
    role: 'assistant' | 'user' | 'tool',
    rawMessage: RawMessageType,
    modelOptions: ModelOptions,
    usage?: UsageStats,
    createdAt?: string,
  ): InternalMessage {
    const lastMessageId =
      this.threadMessages.length > 0
        ? this.threadMessages[this.threadMessages.length - 1].internalMessageId
        : undefined;
    const now = new Date().toISOString();
    createdAt = createdAt || now;
    const newMessage: InternalMessage = {
      internalMessageId: getSeqId(this.threadInfo.threadId, this.threadMessages.length + 1),
      model: modelOptions.model,
      apiMode: modelOptions.apiMode!,
      baseUrl: modelOptions.baseUrl,
      rawMessage: rawMessage,
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
    this.threadInfo.totalUsage = UsageStats.sum(this.threadInfo.totalUsage, usage);
    this.session.totalUsage = UsageStats.sum(this.session.totalUsage, usage);
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
function createSession(
  options: { modelOptions: ModelOptions; holder?: string; parentSessionId?: string },
): Session {
  const sessionId = createSessionId();
  const now = new Date().toISOString();
  const session: Session = {
    sessionId,
    createdAt: now,
    updatedAt: now,
    holder: options.holder,
    parentSessionId: options.parentSessionId,
    messages: [],
    threads: [],
    defaultThread: getSeqId(sessionId, 1),
    totalUsage: UsageStats.empty(),
  };
  forkThread(session, undefined, false, options.modelOptions);
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
  fromMessageId: string | undefined,
  includeHistory: boolean,
  modelOptions: ModelOptions,
): SessionThread {
  const threadId = getSeqId(session.sessionId, session.threads.length + 1);
  const threadInfo: SessionThreadInfo = {
    threadId: threadId,
    rootMessageId: includeHistory ? undefined : fromMessageId,
    headMessageId: fromMessageId,
    modelOptions: modelOptions,
    totalUsage: UsageStats.empty(),
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  session.threads.push(threadInfo);
  const thread = getThreadsFromSession(session, threadInfo.threadId) as SessionThread;
  return thread;
}
