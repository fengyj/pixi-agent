/**
 * Terminal-friendly markdown rendering with ASCII table support.
 *
 * Uses `marked` + `marked-terminal` as the base renderer and extends it
 * with a custom table implementation that wraps columns to fit the
 * available terminal width.
 */

import * as marked from 'marked';
import { markedTerminal } from 'marked-terminal';

import type { ChatLine } from '../types';
import { DISPLAY_ROLE_LABELS } from '../constants';
import { visibleLength, wrapAnsiText, padAnsi } from '../utils/ansi';
import { createRoleDivider } from '../utils/format';

// ── Table rendering ──────────────────────────────────────────────────────────

function renderTableCell(cell: unknown, renderer: marked.Renderer): string {
  if (typeof cell === 'object' && cell !== null) {
    // @ts-expect-error marked parser types — token objects carry `.tokens`
    return renderer.parser.parseInline((cell as any).tokens);
  }
  return String(cell ?? '');
}

/**
 * Render a marked table token as a box-drawn ASCII table that fits within
 * the given column width, wrapping cell content as needed.
 */
function renderAsciiTable(
  token: any,
  width: number,
  renderer: marked.Renderer,
): string {
  const headerCells: string[] = token.header.map((cell: unknown) =>
    renderTableCell(cell, renderer),
  );
  const bodyRows: string[][] = token.rows.map((row: unknown[]) =>
    row.map((cell) => renderTableCell(cell, renderer)),
  );
  const cols = Math.max(headerCells.length, ...bodyRows.map((row) => row.length));

  // Measure desired widths per column.
  const allRows = [headerCells, ...bodyRows];
  const desiredWidths = new Array<number>(cols).fill(0);
  allRows.forEach((row) => {
    row.forEach((cell, index) => {
      desiredWidths[index] = Math.max(desiredWidths[index] ?? 0, visibleLength(cell));
    });
  });

  // Distribute available width proportionally.
  const totalPadding = cols * 2;
  const totalSeparators = cols + 1;
  const available = Math.max(10, width - totalPadding - totalSeparators);
  const totalDesired = desiredWidths.reduce((sum, w) => sum + w, 0) || cols * 10;

  const colWidths = desiredWidths.map((desired) =>
    Math.max(4, Math.min(desired, Math.floor((desired * available) / totalDesired))),
  );

  let allocated = colWidths.reduce((sum, v) => sum + v, 0);

  // Shrink widest columns if over-allocated due to rounding.
  while (allocated > available) {
    const largestIndex = colWidths.reduce(
      (best, w, index) => (w > colWidths[best] ? index : best),
      0,
    );
    if (colWidths[largestIndex] <= 4) break;
    colWidths[largestIndex] -= 1;
    allocated -= 1;
  }

  // Distribute remaining space to the narrowest columns.
  let remaining = available - allocated;
  while (remaining > 0) {
    const smallestIndex = colWidths.reduce(
      (best, w, index) => (w < colWidths[best] ? index : best),
      0,
    );
    colWidths[smallestIndex] += 1;
    remaining -= 1;
  }

  // Wrap each cell's content to its allocated column width.
  const wrapRow = (row: string[]): string[][] => {
    const cells = new Array<string>(cols).fill('');
    row.forEach((cell, index) => {
      cells[index] = cell;
    });
    return cells.map((cell, index) => wrapAnsiText(cell, colWidths[index]));
  };

  const headerWrapped = wrapRow(headerCells);
  const bodyWrapped = bodyRows.map(wrapRow);

  const rowHeights = [headerWrapped, ...bodyWrapped].map((row) =>
    Math.max(...row.map((lines) => lines.length)),
  );

  // Draw box-drawing borders.
  const drawBorder = (left: string, mid: string, right: string) =>
    left + colWidths.map((w) => '─'.repeat(w + 2)).join(mid) + right;

  const lines: string[] = [];
  lines.push(drawBorder('┌', '┬', '┐'));

  const renderRow = (row: string[][], height: number) => {
    for (let lineIndex = 0; lineIndex < height; lineIndex += 1) {
      const cells = row.map((cellLines, colIndex) =>
        padAnsi(cellLines[lineIndex] ?? '', colWidths[colIndex]),
      );
      lines.push('│ ' + cells.join(' │ ') + ' │');
    }
  };

  renderRow(headerWrapped, rowHeights[0]);
  lines.push(drawBorder('├', '┼', '┤'));
  bodyWrapped.forEach((wrapped, idx) => {
    renderRow(wrapped, rowHeights[idx + 1]);
    if (idx < bodyWrapped.length - 1) {
      lines.push(drawBorder('├', '┼', '┤'));
    }
  });
  lines.push(drawBorder('└', '┴', '┘'));

  return lines.join('\n');
}

// ── Markdown → terminal string ───────────────────────────────────────────────

/**
 * Render a markdown string to a terminal-formatted string with ANSI colors,
 * including custom table rendering.
 */
export function renderMarkdown(content: string, width: number): string {
  const markdown = new marked.Marked({
    gfm: true,
    headerIds: false,
  });

  markdown.use(
    markedTerminal({
      width,
      reflowText: true,
      showSectionPrefix: false,
    }),
  );

  markdown.use({
    renderer: {
      table(token: any) {
        return renderAsciiTable(token, width, this as marked.Renderer) + '\n';
      },
      listitem(text: any) {
        if (typeof text === 'object') {
          const item = text;
          const isTask = item.task === true;
          const checkbox = isTask ? (item.checked ? '[x] ' : '[ ] ') : '';
          const tokens =
            isTask && item.tokens?.length && item.tokens[0].type === 'checkbox'
              ? item.tokens.slice(1)
              : item.tokens;
          const innerContent = tokens
            ? this.parser.parse(tokens, !!item.loose)
            : '';
          return '\n* ' + checkbox + innerContent;
        }
        return '\n* ' + text;
      },
    },
  });

  return (markdown.parse(content) as string).trimEnd();
}

// ── Message content composition ──────────────────────────────────────────────

/**
 * Render an array of chat lines into a single terminal-formatted string,
 * separated by role dividers.
 */
export function renderMessageContent(
  lines: ChatLine[],
  dividerWidth: number,
  markdownWidth: number,
): string {
  const visibleLines = lines.filter((line) => line.role !== 'system');

  if (visibleLines.length === 0) {
    return 'No history yet. Send a message to begin the conversation.';
  }

  return visibleLines
    .map((line) => {
      const label =
        line.role === 'user'
          ? DISPLAY_ROLE_LABELS.user
          : DISPLAY_ROLE_LABELS.assistant;
      return `${createRoleDivider(label, dividerWidth)}\n${renderMarkdown(line.content, markdownWidth)}`;
    })
    .join('\n\n');
}
