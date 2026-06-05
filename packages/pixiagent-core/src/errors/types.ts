
export interface PixiAgentErrorOptions {
  code?: string;
  cause?: unknown;
  meta?: Record<string, unknown>;
}

/**
 * Base class for all PixiAgent errors.
 * Ensures `name` is set correctly so OTel `error.type` and `instanceof` checks work reliably.
 */
export class PixiAgentError extends Error {
  public readonly meta?: Record<string, unknown>;
  public readonly code?: string;
  public readonly cause?: unknown;
  constructor(message: string, options?: PixiAgentErrorOptions) {
    super(message);
    // Maintains proper prototype chain in transpiled ES5 output
    Object.setPrototypeOf(this, new.target.prototype);
    this.name = this.constructor.name;
    this.code = options?.code;
    this.cause = options?.cause;
    this.meta = options?.meta;

    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      meta: this.meta,
      cause: this.cause instanceof Error ? this.cause.message : this.cause,
      stack: this.stack,
    };
  } 
}

/**
 * Base error for all retriable failures in PixiAgent.
 *
 * Use this as the parent class for any transient error that callers can retry,
 * e.g. timeout, temporary upstream overload, or short-lived network failures.
 */
export class PixiAgentRetriableError extends PixiAgentError {
  constructor(message: string, options?: PixiAgentErrorOptions) {
    super(message, {
      ...options,
      code: options?.code ?? 'RETRIABLE_ERROR',
    });
  }
}
