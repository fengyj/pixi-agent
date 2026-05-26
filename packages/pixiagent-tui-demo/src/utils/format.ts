/**
 * Formatting utilities for token counts, usage summaries, and role dividers.
 */

import type { UsageSnapshot } from '../types';
import { ROLE_LABEL_WIDTH } from '../constants';

// ── Numeric helpers ──────────────────────────────────────────────────────────

/** Clamp a number between min and max (inclusive). */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

// ── Token formatting ─────────────────────────────────────────────────────────

/**
 * Format a token count into a compact, right-aligned 6-character string.
 *
 * Examples: `"   123"`, `" 1,234"`, `" 12.3k"`, `"  100k"`, `" 1.25m"`
 */
export function formatCompactToken(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2).replace(/\.?0+$/, '')}m`.padStart(6);
  }
  if (value >= 100_000) {
    return `${Math.round(value / 1_000)}k`.padStart(6);
  }
  if (value >= 10_000) {
    return `${(value / 1_000).toFixed(1).replace(/\.0$/, '')}k`.padStart(6);
  }
  return value.toLocaleString('en-US').padStart(6);
}

/**
 * Format a usage snapshot as a compact `I | O | C | T` token summary.
 */
export function formatUsageTokens(usage: UsageSnapshot): string {
  return [
    formatCompactToken(usage.inputTokens),
    formatCompactToken(usage.outputTokens),
    formatCompactToken(usage.cacheReadTokens),
    formatCompactToken(usage.totalTokens),
  ].join(' | ');
}

// ── Role divider ─────────────────────────────────────────────────────────────

/**
 * Create a centered divider line like `────[ USER  ]────` to separate
 * chat messages by role.
 */
export function createRoleDivider(label: string, width: number): string {
  const dividerChar = '─';
  const core = `[ ${label.padEnd(ROLE_LABEL_WIDTH, ' ')} ]`;

  if (width <= core.length + 2) {
    return `${dividerChar}${core}${dividerChar}`;
  }

  const remaining = width - core.length;
  const left = Math.floor(remaining / 2);
  const right = remaining - left;
  return `${dividerChar.repeat(left)}${core}${dividerChar.repeat(right)}`;
}
