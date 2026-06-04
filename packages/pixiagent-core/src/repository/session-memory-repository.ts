import { Session, SessionThreadInfo, MediaSourceInfo, InternalMessage } from '../';
import type { SessionRepository } from './session-repository';
import { PaginationResult } from './utils';
import { nanoid } from 'nanoid';

export class SessionMemoryRepository implements SessionRepository {
  private sessions: Map<string, Session> = new Map();

  async create(session: Omit<Session, 'sessionId'> & { sessionId?: string }): Promise<Session> {
    let sessionId = session.sessionId ?? nanoid(8);
    while (this.sessions.has(sessionId)) {
      sessionId = nanoid(8);
    }

    const nextSession: Session = {
      ...session,
      sessionId,
      messages: [...session.messages],
      threads: [...session.threads],
      mediaSources: session.mediaSources ? { ...session.mediaSources } : undefined,
    };

    this.sessions.set(sessionId, nextSession);
    return nextSession;
  }

  async get<K extends Exclude<keyof Session, 'sessionId'> = Exclude<keyof Session, 'sessionId'>>(
    sessionId: string,
    properties?: readonly K[],
  ): Promise<(Pick<Session, K> & { sessionId: string }) | null> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return null;
    }

    return this.pickSession(session, properties);
  }

  async list<K extends Exclude<keyof Session, 'sessionId'> = Exclude<keyof Session, 'sessionId'>>(
    page?: number,
    pageSize?: number,
    properties?: readonly K[],
  ): Promise<PaginationResult<Pick<Session, K> & { sessionId: string }>> {
    const allSessions = Array.from(this.sessions.values());
    const totalCount = allSessions.length;
    const effectivePage = page && page > 0 ? page : 1;
    const effectivePageSize = pageSize && pageSize > 0 ? pageSize : totalCount;
    const startIndex = (effectivePage - 1) * effectivePageSize;
    const endIndex = pageSize ? startIndex + effectivePageSize : totalCount;

    return {
      items: allSessions.slice(startIndex, endIndex).map((session) => this.pickSession(session, properties)),
      totalCount,
      page: effectivePage,
      pageSize: effectivePageSize,
      endOfPage: endIndex >= totalCount,
    };
  }

  async occupy(sessionId: string, holder: string, timeout?: number): Promise<boolean> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return false;
    }

    const deadline = timeout && timeout > 0 ? Date.now() + timeout : Date.now();

    while (true) {
      if (!session.holder || session.holder === holder) {
        session.holder = holder;
        session.updatedAt = new Date().toISOString();
        return true;
      }

      if (Date.now() >= deadline) {
        return false;
      }

      await this.sleep(100);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async release(sessionId: string, holder: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session || session.holder !== holder) {
      return;
    }
    session.holder = null;
    session.updatedAt = new Date().toISOString();
  }

  async update(session: Partial<Omit<Session, 'sessionId'>> & { sessionId: string }): Promise<void> {
    const existing = this.sessions.get(session.sessionId);
    if (!existing) {
      throw new Error(`Session ${session.sessionId} not found`);
    }

    const rest: Partial<Session> = { ...session };
    delete rest.sessionId;
    Object.assign(existing, rest);
    existing.updatedAt = rest.updatedAt ?? new Date().toISOString();
  }

  async delete(sessionId: string): Promise<void> {
    this.sessions.delete(sessionId);
  }

  async createThread(
    sessionId: string,
    threadInfo: SessionThreadInfo,
    defaultThread?: boolean,
  ): Promise<void> {
    const session = this.getSessionOrThrow(sessionId);
    if (session.threads.some((thread) => thread.threadId === threadInfo.threadId)) {
      throw new Error(`Thread ${threadInfo.threadId} already exists`);
    }

    session.threads.push({ ...threadInfo });
    if (defaultThread) {
      session.defaultThread = threadInfo.threadId;
    }
    session.updatedAt = new Date().toISOString();
  }

  async updateThread(
    sessionId: string,
    threadInfo: Omit<
      SessionThreadInfo,
      'rootMessageId' | 'headMessageId' | 'forkedFromMessageId' | 'createdAt'
    > & { threadId: string; headMessageId: string },
  ): Promise<void> {
    const session = this.getSessionOrThrow(sessionId);
    const thread = session.threads.find((t) => t.threadId === threadInfo.threadId);
    if (!thread) {
      throw new Error(`Thread ${threadInfo.threadId} not found`);
    }

    Object.assign(thread, threadInfo);
    thread.updatedAt = threadInfo.updatedAt ?? new Date().toISOString();
    session.updatedAt = new Date().toISOString();
  }

  async resetThreadHeadMessageId(
    sessionId: string,
    threadId: string,
    headMessageId?: string,
    threadsToDelete?: string[],
  ): Promise<void> {
    const session = this.getSessionOrThrow(sessionId);
    const thread = session.threads.find((item) => item.threadId === threadId);
    if (!thread) {
      throw new Error(`Thread ${threadId} not found`);
    }

    if (threadsToDelete?.includes(session.defaultThread)) {
      throw new Error('Cannot delete the default thread');
    }

    thread.headMessageId = headMessageId;
    thread.updatedAt = new Date().toISOString();

    if (threadsToDelete?.length) {
      session.threads = session.threads.filter((item) => !threadsToDelete.includes(item.threadId));
    }

    session.updatedAt = new Date().toISOString();
  }

  async deleteThread(sessionId: string, threadId: string): Promise<void> {
    const session = this.getSessionOrThrow(sessionId);
    if (session.defaultThread === threadId) {
      throw new Error('Cannot delete the default thread');
    }

    const index = session.threads.findIndex((thread) => thread.threadId === threadId);
    if (index === -1) {
      return;
    }

    session.threads.splice(index, 1);
    session.updatedAt = new Date().toISOString();
  }

  async getThreads(sessionId: string): Promise<SessionThreadInfo[]> {
    const session = this.sessions.get(sessionId);
    return session ? [...session.threads] : [];
  }

  async getMessages(sessionId: string, threadId?: string): Promise<InternalMessage[]> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      return [];
    }
    if (!threadId) {
      return [...session.messages];
    }

    const thread = session.threads.find((item) => item.threadId === threadId);
    if (!thread || !thread.headMessageId) {
      return [];
    }

    const messagesById = new Map(session.messages.map((message) => [message.internalMessageId, message]));
    const result: InternalMessage[] = [];
    let currentMessageId: string | undefined = thread.headMessageId;

    while (currentMessageId) {
      const message = messagesById.get(currentMessageId);
      if (!message) {
        break;
      }
      result.unshift(message);
      if (currentMessageId === thread.rootMessageId) {
        break;
      }
      currentMessageId = message.previousMessageId;
    }

    return result;
  }

  async getMediaSources(sessionId: string): Promise<MediaSourceInfo[]> {
    const session = this.sessions.get(sessionId);
    if (!session || !session.mediaSources) {
      return [];
    }
    return Object.values(session.mediaSources);
  }

  async addMediaSource(sessionId: string, mediaSource: MediaSourceInfo): Promise<void> {
    const session = this.getSessionOrThrow(sessionId);
    session.mediaSources = {
      ...(session.mediaSources ?? {}),
      [mediaSource.mediaSourceId]: mediaSource,
    };
    session.updatedAt = new Date().toISOString();
  }

  async deleteMediaSource(sessionId: string, mediaSourceIds: string | string[]): Promise<string[]> {
    const session = this.getSessionOrThrow(sessionId);
    if (!session.mediaSources) {
      return [];
    }
    const ids = Array.isArray(mediaSourceIds) ? mediaSourceIds : [mediaSourceIds];
    const deleted: string[] = [];
    for (const id of ids) {
      if (id in session.mediaSources) {
        deleted.push(id);
        delete session.mediaSources[id];
      }
    }
    session.updatedAt = new Date().toISOString();
    return deleted;
  }

  async patchWithNewMessage(
    sessionInfo: Partial<Omit<Session, 'sessionId' | 'messages' | 'threads'>> & {
      sessionId: string;
    },
    threadInfo: Partial<
      Omit<SessionThreadInfo, 'rootMessageId' | 'createdAt'> & {
        threadId: string;
        headMessageId: string;
      }
    >,
    message: InternalMessage,
  ): Promise<void> {
    const session = this.getSessionOrThrow(sessionInfo.sessionId);
    const thread = session.threads.find((item) => item.threadId === threadInfo.threadId);
    if (!thread) {
      throw new Error(`Thread ${threadInfo.threadId} not found`);
    }

    const restSession: Partial<Session> = { ...sessionInfo };
    delete restSession.sessionId;
    Object.assign(session, restSession);
    session.updatedAt = restSession.updatedAt ?? new Date().toISOString();

    Object.assign(thread, threadInfo);
    thread.updatedAt = threadInfo.updatedAt ?? new Date().toISOString();

    session.messages.push(message);
    session.updatedAt = new Date().toISOString();
  }

  private getSessionOrThrow(sessionId: string): Session {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    return session;
  }

  private pickSession<K extends Exclude<keyof Session, 'sessionId'>>(
    session: Session,
    properties?: readonly K[],
  ): Pick<Session, K> & { sessionId: string } {
    if (!properties) {
      return session as Pick<Session, K> & { sessionId: string };
    }

    const result = { sessionId: session.sessionId } as Pick<Session, K> & {
      sessionId: string;
    };

    for (const key of properties) {
      result[key] = session[key] as Session[K];
    }

    return result;
  }
}
