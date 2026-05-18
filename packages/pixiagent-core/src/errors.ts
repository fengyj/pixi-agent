/**
 * Base class for all PixiAgent errors.
 * Ensures `name` is set correctly so OTel `error.type` and `instanceof` checks work reliably.
 */
export class PixiAgentError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = this.constructor.name;
    // Maintains proper prototype chain in transpiled ES5 output
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// ─── Agent lifecycle errors ───────────────────────────────────────────────────

/**
 * Thrown when the agent's input queue is full and cannot accept new messages.
 * Callers should treat this as a transient "busy" condition and retry later.
 */
export class InputQueueFullError extends PixiAgentError {
  constructor(public readonly queueCapacity: number) {
    super(
      `The input queue is full (capacity: ${queueCapacity}). Please wait for the agent to process pending messages.`,
    );
  }
}

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
