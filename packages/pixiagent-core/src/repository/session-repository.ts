import { Session, SessionThreadInfo, MediaSourceInfo, InternalMessage } from '../';
import { PaginationResult } from './utils';

export interface SessionRepository {
  /**
   * Creates a new session in the repository.
   * If the sessionId is not provided, a new one will be generated.
   * @param session
   */
  create(session: Omit<Session, 'sessionId'> & { sessionId?: string }): Promise<Session>;

  /**
   * Gets a session from the repository.
   * If properties are provided, only those properties will be returned.
   * @param sessionId
   * @param properties
   */
  get<K extends Exclude<keyof Session, 'sessionId'> = Exclude<keyof Session, 'sessionId'>>(
    sessionId: string,
    properties?: readonly K[],
  ): Promise<(Pick<Session, K> & { sessionId: string }) | null>;

  /**
   * Gets a list of sessions from the repository.
   * The sessions are sorted by updatedAt in descending order.
   * @param page
   * @param pageSize if not provided, all sessions will be returned.
   * @param properties
   */
  list<K extends Exclude<keyof Session, 'sessionId'> = Exclude<keyof Session, 'sessionId'>>(
    page?: number,
    pageSize?: number,
    properties?: readonly K[],
  ): Promise<PaginationResult<Pick<Session, K> & { sessionId: string }>>;

  /**
   * Tries to occupy a session for a specific holder.
   * If the session is already occupied by another holder, it will return false.
   * @param sessionId
   * @param holder
   * @param timeout in milliseconds.
   *                If provided, will retry to occupy the session until the timeout is reached.
   *                If not provided, will try to occupy the session once.
   */
  occupy(sessionId: string, holder: string, timeout?: number): Promise<boolean>;

  /**
   * Releases a session from a specific holder.
   * @param sessionId
   * @param holder
   */
  release(sessionId: string, holder: string): Promise<void>;

  /**
   * Updates a session with the given fields in the repository.
   *
   * @param session
   */
  update(
    session: Partial<Omit<Session, 'sessionId' | 'createdAt'>> & { sessionId: string },
  ): Promise<void>;

  /**
   * Deletes a session from the repository.
   * @param sessionId
   */
  delete(sessionId: string): Promise<void>;

  /**
   * Creates a new thread of the session in the repository.
   * @param sessionId
   * @param threadInfo
   * @param defaultThread
   */
  createThread(
    sessionId: string,
    threadInfo: SessionThreadInfo,
    defaultThread?: boolean,
  ): Promise<void>;

  /**
   * Updates a thread of the session in the repository.
   * Or deletes messages by setting the headMessageId to a previous messageId.
   *
   * When using setting the headMessageId to delete messages, the following rules which are not handled
   * by this function, but should be followed:
   *
   * If a message is deleted, the threads folked from or after the message, should be deleted as well.
   * In another word, the headMessageId shouldn't be the id of the message which is before or equal
   * the forkedFromMessageId of the thread. If so, should use deleteThread instead.
   *
   * @param sessionId
   * @param threadInfo
   */
  updateThread(
    sessionId: string,
    threadInfo: Omit<
      SessionThreadInfo,
      'rootMessageId' | 'headMessageId' | 'forkedFromMessageId' | 'createdAt'
    > & { threadId: string; headMessageId: string },
  ): Promise<void>;

  /**
   * Resets the headMessageId of a thread in the repository.
   * 
   * This function is used to "delete" the messages created after the message of the headMessageId,
   * and the threads which are forked from those messages.
   * @param sessionId 
   * @param threadId 
   * @param headMessageId 
   * @param threadsToDelete 
   */
  resetThreadHeadMessageId(
    sessionId: string,
    threadId: string,
    headMessageId?: string,
    threadsToDelete?: string[],
  ): Promise<void>;

  /**
   * Deletes a thread of the session from the repository.
   * The function shouldn't work on the default thread, and should throw an error if attempted.
   * To "delete" the default thread, should set anther thread as the default one first.
   * @param sessionId
   * @param threadId
   */
  deleteThread(sessionId: string, threadId: string): Promise<void>;

  /**
   * Gets threads of the session from the repository.
   * @param sessionId
   */
  getThreads(sessionId: string): Promise<SessionThreadInfo[]>;

  /**
   * Gets messages of a session from the repository.
   * @param sessionId
   * @param threadId if provided, only messages of the thread will be returned.
   */
  getMessages(sessionId: string, threadId?: string): Promise<InternalMessage[]>;

  /**
   * Gets media sources of a session from the repository.
   * @param sessionId
   */
  getMediaSources(sessionId: string): Promise<MediaSourceInfo[]>;

  /**
   * Adds a media source to a session in the repository.
   * @param sessionId
   * @param mediaSource
   */
  addMediaSource(sessionId: string, mediaSource: MediaSourceInfo): Promise<void>;

  /**
   * Deletes media sources of a session from the repository.
   * @param sessionId
   * @param mediaSourceIds
   */
  deleteMediaSource(sessionId: string, mediaSourceIds: string | string[]): Promise<string[]>;

  /**
   * Patches a session with a new message in the repository.
   * @param sessionInfo only the fields that are provided will be updated. The sessionId is required.
   * @param threadInfo only the fields that are provided will be updated. The threadId and headMessageId are required.
   * @param message
   */
  patchWithNewMessage(
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
  ): Promise<void>;
}
