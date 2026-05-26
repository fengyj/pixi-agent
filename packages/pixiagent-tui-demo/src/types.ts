/**
 * Shared type definitions for the TUI demo application.
 */

/** A single chat message displayed in the conversation history. */
export type ChatLine = {
  /** Unique identifier for stable React list keys. */
  readonly id: string;
  readonly role: 'system' | 'user' | 'assistant';
  readonly content: string;
};

/** Result of parsing a `/command args` input string. */
export type ParsedCommand = {
  readonly name: string;
  readonly args: string;
};

/** Subset of core UsageStats used for UI display. */
export type UsageSnapshot = {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly cacheReadTokens: number;
};
