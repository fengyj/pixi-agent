/**
 * Slash-command parsing for TUI input.
 */

import type { ParsedCommand } from '../types';

/**
 * Parse a `/command args` string into its name and arguments.
 *
 * Returns `null` if the input is not a valid command (doesn't start with `/`
 * or has an empty command name).
 *
 * @example
 * ```ts
 * parseCommand('/exit')        // { name: 'exit', args: '' }
 * parseCommand('/cancel oops') // { name: 'cancel', args: 'oops' }
 * parseCommand('hello')        // null
 * parseCommand('/')            // null
 * ```
 */
export function parseCommand(input: string): ParsedCommand | null {
  if (!input.startsWith('/')) return null;

  const body = input.slice(1);
  if (body.length === 0) return null;

  const firstWhitespaceIndex = body.search(/\s/);
  if (firstWhitespaceIndex === -1) {
    return { name: body.toLowerCase(), args: '' };
  }

  return {
    name: body.slice(0, firstWhitespaceIndex).toLowerCase(),
    args: body.slice(firstWhitespaceIndex).trimStart(),
  };
}
