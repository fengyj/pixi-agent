/**
 * Base class for all PixiAgent errors.
 * Ensures `name` is set correctly so OTel `error.type` and `instanceof` checks work reliably.
 */
export class PixiAgentError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
    // Maintains proper prototype chain in transpiled ES5 output
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Agent lifecycle errors ───────────────────────────────────────────────────

/**
 * Thrown when the agent has exceeded its configured maximum iteration limit.
 * Indicates the agent loop ran out of budget, not that an error occurred.
 */
export class MaxIterationsExceededError extends PixiAgentError {
  constructor(public readonly maxIterations: number) {
    super(`The agent has reached the maximum number of iterations (${maxIterations}).`);
  }
}

/**
 * Thrown (or used as the abort reason) when the agent execution is intentionally interrupted.
 */
export class AgentInterruptedError extends PixiAgentError {
  constructor(public readonly reason: string = 'User interrupted') {
    super(`Agent execution was interrupted: ${reason}`);
  }
}

/**
 * Base error for all retriable failures in PixiAgent.
 *
 * Use this as the parent class for any transient error that callers can retry,
 * e.g. timeout, temporary upstream overload, or short-lived network failures.
 */
export class PixiAgentRetriableError extends PixiAgentError {
  constructor(message: string, cause?: unknown) {
    super(message);
    if (cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = cause;
    }
  }
}

/**
 * Base error for timeout failures in PixiAgent.
 * This is distinct from user-triggered cancellation (interrupt/abort).
 */
export class PixiAgentTimeoutError extends PixiAgentRetriableError {
  constructor(message: string, public readonly timeoutMs?: number, cause?: unknown) {
    super(message, cause);
  }
}

/**
 * Thrown when an upstream model provider request times out.
 */
export class ModelRequestTimeoutError extends PixiAgentTimeoutError {
  constructor(
    public readonly provider: 'openai' | 'anthropic' | string,
    timeoutMs?: number,
    cause?: unknown,
  ) {
    super(
      timeoutMs !== undefined
        ? `Model request to ${provider} timed out after ${timeoutMs}ms.`
        : `Model request to ${provider} timed out.`,
      timeoutMs,
      cause,
    );
  }
}

// ─── Configuration errors ─────────────────────────────────────────────────────

/**
 * Thrown when the API mode and base URL cannot be resolved for the given model/options.
 * This is a configuration error and should not occur in normal operation.
 */
export class ApiModeResolutionError extends PixiAgentError {
  constructor(
    public readonly model: string,
    public readonly baseUrl: string | undefined,
  ) {
    super(
      `Cannot resolve the API mode for model "${model}"${baseUrl ? ` with baseUrl "${baseUrl}"` : ''}.`,
    );
  }
}

// ─── Transport / message conversion errors ────────────────────────────────────

/**
 * Thrown when a message with an unsupported or unexpected role is encountered
 * during transport conversion. Indicates a programming error in the caller.
 */
export class InvalidMessageError extends PixiAgentError {
  constructor(message: string) {
    super(message);
  }
}
