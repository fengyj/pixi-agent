import { Session, SessionThreadInfo, MediaSourceInfo, InternalMessage } from '..';
import type { SessionRepository } from './session-repository';
import { PaginationResult } from './utils';
import { promises as fs } from 'fs';
import path from 'path';
import { nanoid } from 'nanoid';

export class SessionFileSystemRepository implements SessionRepository {
  constructor(private repoFolderPath: string) {
    this.repoFolderPath = path.resolve(repoFolderPath);
  }

  async create(session: Omit<Session, 'sessionId'> & { sessionId?: string }): Promise<Session> {
    await this.ensureRepoFolder();

    let sessionId = session.sessionId ?? nanoid(8);
    let sessionFolder = this.getSessionFolder(sessionId);
    while (await this.fileExists(sessionFolder)) {
      sessionId = nanoid(8);
      sessionFolder = this.getSessionFolder(sessionId);
    }

    await fs.mkdir(sessionFolder, { recursive: true });
    const messageFile = this.getMessagesFile(sessionId);
    const nextSession: Session = {
      ...session,
      sessionId,
      messages: [...session.messages],
      threads: [...session.threads],
      mediaSources: session.mediaSources ? { ...session.mediaSources } : undefined,
    };

    await this.saveSessionMetadata(nextSession);
    await fs.writeFile(messageFile, '', 'utf-8');
    if (nextSession.messages.length > 0) {
      await this.writeMessages(sessionId, nextSession.messages);
    }

    return nextSession;
  }

  async get<K extends Exclude<keyof Session, 'sessionId'> = Exclude<keyof Session, 'sessionId'>>(
    sessionId: string,
    properties?: readonly K[],
  ): Promise<(Pick<Session, K> & { sessionId: string }) | null> {
    const loadMessages = !properties || properties.includes('messages' as K);
    const session = await this.loadSession(sessionId, loadMessages);
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
    await this.ensureRepoFolder();
    const entries = await fs.readdir(this.repoFolderPath, { withFileTypes: true });
    const sessionStats = await Promise.all(
      entries
        .filter((entry) => entry.isDirectory())
        .map(async (entry) => {
          const sessionId = entry.name;
          const sessionFile = this.getSessionFile(sessionId);
          if (!(await this.fileExists(sessionFile))) {
            return null;
          }
          const stat = await fs.stat(sessionFile);
          return { sessionId, mtime: stat.mtimeMs };
        }),
    );

    const sortedSessionIds = sessionStats
      .filter((info): info is { sessionId: string; mtime: number } => info !== null)
      .sort((a, b) => b.mtime - a.mtime)
      .map((info) => info.sessionId);

    const totalCount = sortedSessionIds.length;
    const effectivePage = page && page > 0 ? page : 1;
    const effectivePageSize = pageSize && pageSize > 0 ? pageSize : totalCount;
    const startIndex = (effectivePage - 1) * effectivePageSize;
    const endIndex = pageSize ? startIndex + effectivePageSize : totalCount;
    const selected = sortedSessionIds.slice(startIndex, endIndex);
    const loadMessages = !properties || properties.includes('messages' as K);

    const items = [] as Array<Pick<Session, K> & { sessionId: string }>;
    for (const sessionId of selected) {
      const session = await this.loadSession(sessionId, loadMessages);
      if (!session) {
        continue;
      }
      items.push(this.pickSession(session, properties));
    }

    return {
      items,
      totalCount,
      page: effectivePage,
      pageSize: effectivePageSize,
      endOfPage: endIndex >= totalCount,
    };
  }

  async occupy(sessionId: string, holder: string, timeout?: number): Promise<boolean> {
    const sessionFolder = this.getSessionFolder(sessionId);
    if (!(await this.fileExists(sessionFolder))) {
      return false;
    }

    const lockFile = this.getLockFile(sessionId);
    const deadline = timeout && timeout > 0 ? Date.now() + timeout : undefined;

    while (true) {
      if (!(await this.fileExists(lockFile))) {
        await fs.writeFile(lockFile, JSON.stringify({ holder, acquiredAt: new Date().toISOString() }, null, 2), 'utf-8');
        return true;
      }

      const raw = JSON.parse(await fs.readFile(lockFile, 'utf-8')) as { holder: string };
      if (raw.holder === holder) {
        return true;
      }

      if (deadline === undefined || Date.now() >= deadline) {
        return false;
      }

      await this.sleep(200);
    }
  }

  async release(sessionId: string, holder: string): Promise<void> {
    const lockFile = this.getLockFile(sessionId);
    if (!(await this.fileExists(lockFile))) {
      return;
    }

    const raw = JSON.parse(await fs.readFile(lockFile, 'utf-8')) as { holder: string };
    if (raw.holder !== holder) {
      return;
    }

    await fs.rm(lockFile, { force: true });
  }

  async update(session: Partial<Omit<Session, 'sessionId'>> & { sessionId: string }): Promise<void> {
    const existing = await this.loadSession(session.sessionId, false);
    if (!existing) {
      throw new Error(`Session ${session.sessionId} not found`);
    }

    const updated: Session = {
      ...existing,
      ...session,
      messages: existing.messages,
      threads: existing.threads,
      mediaSources: existing.mediaSources,
    };

    if (session.messages) {
      await this.writeMessages(session.sessionId, session.messages);
      updated.messages = [...session.messages];
    }

    if (session.threads) {
      updated.threads = [...session.threads];
    }

    await this.saveSessionMetadata(updated);
  }

  async delete(sessionId: string): Promise<void> {
    const sessionFolder = this.getSessionFolder(sessionId);
    await fs.rm(sessionFolder, { recursive: true, force: true });
  }

  async createThread(
    sessionId: string,
    threadInfo: SessionThreadInfo,
    defaultThread?: boolean,
  ): Promise<void> {
    const session = await this.loadSession(sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    if (session.threads.some((thread) => thread.threadId === threadInfo.threadId)) {
      throw new Error(`Thread ${threadInfo.threadId} already exists`);
    }

    session.threads.push({ ...threadInfo });
    if (defaultThread) {
      session.defaultThread = threadInfo.threadId;
    }
    session.updatedAt = new Date().toISOString();
    await this.saveSessionMetadata(session);
  }

  async updateThread(
    sessionId: string,
    threadInfo: Omit<
      SessionThreadInfo,
      'rootMessageId' | 'headMessageId' | 'forkedFromMessageId' | 'createdAt'
    > & { threadId: string; headMessageId: string },
  ): Promise<void> {
    const session = await this.loadSession(sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    const thread = session.threads.find((item) => item.threadId === threadInfo.threadId);
    if (!thread) {
      throw new Error(`Thread ${threadInfo.threadId} not found`);
    }

    Object.assign(thread, threadInfo);
    thread.updatedAt = threadInfo.updatedAt ?? new Date().toISOString();
    session.updatedAt = new Date().toISOString();
    await this.saveSessionMetadata(session);
  }

  async resetThreadHeadMessageId(
    sessionId: string,
    threadId: string,
    headMessageId?: string,
    threadsToDelete?: string[],
  ): Promise<void> {
    const session = await this.loadSession(sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

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
    await this.saveSessionMetadata(session);
  }

  async deleteThread(sessionId: string, threadId: string): Promise<void> {
    const session = await this.loadSession(sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    if (session.defaultThread === threadId) {
      throw new Error('Cannot delete the default thread');
    }

    const index = session.threads.findIndex((thread) => thread.threadId === threadId);
    if (index === -1) {
      return;
    }

    session.threads.splice(index, 1);
    session.updatedAt = new Date().toISOString();
    await this.saveSessionMetadata(session);
  }

  async getThreads(sessionId: string): Promise<SessionThreadInfo[]> {
    const session = await this.loadSession(sessionId, false);
    return session ? [...session.threads] : [];
  }

  async getMessages(sessionId: string, threadId?: string): Promise<InternalMessage[]> {
    const messages = await this.readMessages(sessionId);
    if (!threadId) {
      return messages;
    }

    const session = await this.loadSession(sessionId, false);
    if (!session) {
      return [];
    }

    const thread = session.threads.find((item) => item.threadId === threadId);
    if (!thread || !thread.headMessageId) {
      return [];
    }

    const messagesById = new Map(messages.map((message) => [message.internalMessageId, message]));
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
    const session = await this.loadSession(sessionId, false);
    if (!session || !session.mediaSources) {
      return [];
    }
    return Object.values(session.mediaSources);
  }

  async addMediaSource(sessionId: string, mediaSource: MediaSourceInfo): Promise<void> {
    const session = await this.loadSession(sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    session.mediaSources = {
      ...(session.mediaSources ?? {}),
      [mediaSource.mediaSourceId]: mediaSource,
    };
    session.updatedAt = new Date().toISOString();
    await this.saveSessionMetadata(session);
  }

  async deleteMediaSource(sessionId: string, mediaSourceIds: string | string[]): Promise<string[]> {
    const session = await this.loadSession(sessionId, false);
    if (!session || !session.mediaSources) {
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
    await this.saveSessionMetadata(session);
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
    const session = await this.loadSession(sessionInfo.sessionId, false);
    if (!session) {
      throw new Error(`Session ${sessionInfo.sessionId} not found`);
    }

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

    await this.appendMessage(sessionInfo.sessionId, message);
    session.updatedAt = new Date().toISOString();
    await this.saveSessionMetadata(session);
  }

  private async ensureRepoFolder(): Promise<void> {
    await fs.mkdir(this.repoFolderPath, { recursive: true });
  }

  private getSessionFolder(sessionId: string): string {
    return path.join(this.repoFolderPath, sessionId);
  }

  private getSessionFile(sessionId: string): string {
    return path.join(this.getSessionFolder(sessionId), 'session.json');
  }

  private getMessagesFile(sessionId: string): string {
    return path.join(this.getSessionFolder(sessionId), 'messages.jsonl');
  }

  private getLockFile(sessionId: string): string {
    return path.join(this.getSessionFolder(sessionId), 'lock.json');
  }

  private async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  private async loadSession(sessionId: string, loadMessages: boolean): Promise<Session | null> {
    const file = this.getSessionFile(sessionId);
    if (!(await this.fileExists(file))) {
      return null;
    }

    const raw = JSON.parse(await fs.readFile(file, 'utf-8')) as Omit<Session, 'messages'> & { messages?: InternalMessage[] };
    const session: Session = {
      ...raw,
      sessionId,
      messages: [],
      threads: raw.threads ?? [],
      mediaSources: raw.mediaSources ?? undefined,
    } as Session;

    if (loadMessages) {
      session.messages = await this.readMessages(sessionId);
    }

    return session;
  }

  private async saveSessionMetadata(session: Session): Promise<void> {
    const sessionCopy = { ...session } as Record<string, unknown>;
    delete sessionCopy.sessionId;
    delete sessionCopy.messages;
    const file = this.getSessionFile(session.sessionId);
    await fs.writeFile(file, JSON.stringify(sessionCopy, null, 2), 'utf-8');
  }

  private async appendMessage(sessionId: string, message: InternalMessage): Promise<void> {
    const file = this.getMessagesFile(sessionId);
    await fs.appendFile(file, JSON.stringify(message) + '\n', 'utf-8');
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

  private async writeMessages(sessionId: string, messages: InternalMessage[]): Promise<void> {
    const file = this.getMessagesFile(sessionId);
    const content = messages.map((message) => JSON.stringify(message)).join('\n');
    await fs.writeFile(file, content.length > 0 ? content + '\n' : '', 'utf-8');
  }

  private async readMessages(sessionId: string): Promise<InternalMessage[]> {
    const file = this.getMessagesFile(sessionId);
    if (!(await this.fileExists(file))) {
      return [];
    }

    const content = await fs.readFile(file, 'utf-8');
    const lines = content.split('\n').filter((line) => line.trim().length > 0);
    const messages = new Map<string, InternalMessage>();
    const order: string[] = [];

    for (const line of lines) {
      try {
        const item = JSON.parse(line) as { internalMessageId: string; deleted?: boolean } & InternalMessage;
        if (item.deleted) {
          messages.delete(item.internalMessageId);
          const index = order.indexOf(item.internalMessageId);
          if (index !== -1) {
            order.splice(index, 1);
          }
          continue;
        }

        if (!messages.has(item.internalMessageId)) {
          messages.set(item.internalMessageId, item);
          order.push(item.internalMessageId);
        }
      } catch {
        // ignore invalid lines
      }
    }

    return order.map((id) => messages.get(id)!) as InternalMessage[];
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
