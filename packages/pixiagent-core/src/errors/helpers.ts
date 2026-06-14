import { PixiAgentError, PixiAgentRetriableError } from './types';
import type { RoleType } from '../message';

export const PixiAgentErrorBuilder = {
  agentMaxIterationsExceeded: (maxIterations: number): PixiAgentError => {
    return new PixiAgentError(
      `The agent has reached the maximum number of iterations (${maxIterations}).`,
      {
        meta: { maxIterations },
        code: 'AGENT_MAX_ITERATIONS_EXCEEDED',
      },
    );
  },
  agentInterrupted: (reason: string = 'User interrupted'): PixiAgentError => {
    return new PixiAgentError(`Agent execution was interrupted: ${reason}`, {
      meta: { reason },
      code: 'AGENT_INTERRUPTED',
    });
  },
  agentConcurrentExecution: (): PixiAgentError => {
    return new PixiAgentError(
      'The agent is already running and does not allow concurrent execute calls.',
      {
        code: 'AGENT_CONCURRENT_EXECUTION',
      },
    );
  },
  invalidMessage: (
    message: string,
    messageRole?: RoleType,
    detailCode?: string,
    cause?: unknown,
  ): PixiAgentError => {
    return new PixiAgentError(message, {
      code: 'INVALID_MESSAGE',
      cause: cause,
      meta: { detailCode, messageRole },
    });
  },
  apiModeResolutionFailed: (model: string, baseUrl?: string): PixiAgentError => {
    return new PixiAgentError(
      `Cannot resolve the API mode for model "${model}"${baseUrl ? ` with baseUrl "${baseUrl}"` : ''}.`,
      {
        code: 'API_MODE_RESOLUTION_FAILED',
        meta: { model, baseUrl },
      },
    );
  },
  modelResponseError: (
    message: string,
    baseUrl?: string,
    detailCode?: string,
    cause?: unknown,
  ): PixiAgentRetriableError => {
    return new PixiAgentRetriableError(message, {
      code: 'MODEL_RESPONSE_ERROR',
      cause: cause,
      meta: { detailCode },
    });
  },
  modelRequestTimeout: (
    baseUrl: string,
    timeoutMs?: number,
    detailCode?: string,
    cause?: unknown,
  ): PixiAgentRetriableError => {
    return new PixiAgentRetriableError(
      timeoutMs !== undefined
        ? `Model request to ${baseUrl} timed out after ${timeoutMs}ms.`
        : `Model request to ${baseUrl} timed out.`,
      {
        code: 'MODEL_REQUEST_TIMEOUT',
        cause: cause,
        meta: { baseUrl, detailCode, timeoutMs },
      },
    );
  },
  modelRequestRetriableError: (
    message: string,
    baseUrl: string,
    detailCode?: string,
    cause?: unknown,
  ): PixiAgentRetriableError => {
    return new PixiAgentRetriableError(message, {
      cause: cause,
      meta: { detailCode, baseUrl },
      code: 'MODEL_REQUEST_RETRIABLE_ERROR',
    });
  },
  threadNotFound: (threadId: string): PixiAgentError => {
    return new PixiAgentError(`Thread with id "${threadId}" not found.`, {
      code: 'THREAD_NOT_FOUND',
      meta: { threadId },
    });
  },
};
