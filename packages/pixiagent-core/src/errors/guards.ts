import { PixiAgentError, PixiAgentRetriableError } from './types';

const TIMEOUT_ERROR_NAMES = new Set([
  'TimeoutError',
  'RequestTimeoutError',
  'APIConnectionTimeoutError',
]);

const TIMEOUT_ERROR_CODES = new Set([
  'MODEL_REQUEST_TIMEOUT',  // PixiAgentRetriableError
  'ESOCKETTIMEDOUT',
  'UND_ERR_CONNECT_TIMEOUT',
]);

const ABORT_ERROR_NAMES = new Set([
  'AbortError',
  'APIUserAbortError',
]);

const ABORT_ERROR_CODES = new Set([
  'ABORT_ERR',
  'ERR_CANCELED',
  'AGENT_INTERRUPTED',  // PixiAgentErrors.agentInterrupted
]);

const RETRIABLE_ERROR_NAMES = new Set([
  ...TIMEOUT_ERROR_NAMES,
  'PixiAgentRetriableError',
]);

const RETRIABLE_ERROR_CODES = new Set([
  ...TIMEOUT_ERROR_CODES,
  'MODEL_REQUEST_RETRIABLE_ERROR',
]);

/**
 * Heuristic detector for provider SDK timeout errors.
 * We intentionally avoid mapping caller-triggered abort/cancel errors here.
 */
function isLikelyTimeoutError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  const enriched = error as Error & {
    code?: unknown;
    status?: unknown;
    type?: unknown;
  };

  if (TIMEOUT_ERROR_NAMES.has(error.name)) {
    return true;
  }

  if (typeof enriched.code === 'string' && TIMEOUT_ERROR_CODES.has(enriched.code)) {
    return true;
  }

  if (typeof enriched.status === 'number' && enriched.status === 408) {
    return true;
  }

  const typeText = typeof enriched.type === 'string' ? enriched.type.toLowerCase() : '';
  if (typeText.includes('timeout')) {
    return true;
  }

  const message = error.message.toLowerCase();
  return (
    message.includes('timeout') ||
    message.includes('timed out') ||
    message.includes('request timeout') ||
    message.includes('connection timeout')
  );
}

/**
 * Heuristic detector for provider SDK abort/cancel errors.
 * This should be mapped to AgentInterruptedError by upper layers.
 */
function isLikelyAbortError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  if (ABORT_ERROR_NAMES.has(error.name)) {
    return true;
  }

  const enriched = error as Error & {
    code?: unknown;
    type?: unknown;
  };

  if (typeof enriched.code === 'string' && ABORT_ERROR_CODES.has(enriched.code)) {
    return true;
  }

  const typeText = typeof enriched.type === 'string' ? enriched.type.toLowerCase() : '';
  if (typeText.includes('abort') || typeText.includes('cancel')) {
    return true;
  }

  const message = error.message.toLowerCase();
  return (
    message.includes('aborted') ||
    message.includes('request was aborted') ||
    message.includes('cancelled') ||
    message.includes('canceled')
  );
}

/**
 * Type-first detector for retriable errors.
 *
 * Any error that extends PixiAgentRetriableError is retriable.
 * User-triggered interruption is explicitly non-retriable.
 * Falls back to checking nested cause chain for wrapped errors.
 */
function isRetriableError(error: unknown): boolean {

  if (error instanceof PixiAgentRetriableError) {
    return true;
  }

  if (!(error instanceof Error)) {
    return false;
  }

  if(RETRIABLE_ERROR_NAMES.has(error.name)) {
    return true;
  }

  const enriched = error as Error & {
    code?: unknown;
    cause?: unknown;
  };
  
  if(enriched.code && typeof enriched.code === 'string' && RETRIABLE_ERROR_CODES.has(enriched.code)) {
    return true;
  }

  if (enriched.cause !== undefined && enriched.cause !== error) {
    return isRetriableError(enriched.cause);
  }

  return false;
}

function isPixiAgentError(error: unknown): error is Error & { code?: string; meta?: Record<string, unknown> } {
  if(error instanceof PixiAgentError) {
    return true;
  }
  return error instanceof Error && 'code' in error && typeof error.code === 'string';
}

export const ErrorGuards = {
  isLikelyTimeoutError,
  isLikelyAbortError,
  isRetriableError,
  isPixiAgentError,
};