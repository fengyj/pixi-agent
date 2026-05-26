/**
 * Display constants and configuration values used across TUI components.
 */

import type { UsageSnapshot } from './types';

// ── Role display ─────────────────────────────────────────────────────────────

export const DISPLAY_ROLE_LABELS = {
  user: 'USER',
  assistant: 'AGENT',
} as const;

export const ROLE_LABEL_WIDTH = Math.max(
  DISPLAY_ROLE_LABELS.user.length,
  DISPLAY_ROLE_LABELS.assistant.length,
);

// ── Color palette ────────────────────────────────────────────────────────────

/** Catppuccin-inspired color palette for consistent theming. */
export const COLORS = {
  /** Primary accent — header, links. */
  primary: '#89b4fa',
  /** Secondary muted text — hints, help. */
  muted: '#94a3b8',
  /** Warning / status — status bar highlights. */
  warning: '#f9e2af',
  /** Border color for panels. */
  border: '#4b5563',
  /** Outer frame border. */
  frame: '#3a4a5a',
  /** Placeholder text. */
  placeholder: '#6b7280',
} as const;

// ── Usage defaults ───────────────────────────────────────────────────────────

export const EMPTY_USAGE: UsageSnapshot = Object.freeze({
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  cacheReadTokens: 0,
});

// ── Layout ───────────────────────────────────────────────────────────────────

/** Minimum width for rendered content areas. */
export const MIN_CONTENT_WIDTH = 24;

/** Default terminal dimensions when stdout info is unavailable. */
export const DEFAULT_COLUMNS = 120;
export const DEFAULT_ROWS = 40;

/** Number of rows reserved for chrome (header, status, input, borders). */
export const CHROME_ROWS = 12;

/** Minimum height for the message panel viewport. */
export const MIN_MESSAGE_PANEL_HEIGHT = 4;

/** Minimum height for the input panel. */
export const MIN_INPUT_HEIGHT = 3;
