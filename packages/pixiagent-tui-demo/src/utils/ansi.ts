/**
 * ANSI-aware text measurement and wrapping utilities.
 *
 * Handles terminal escape sequences, full-width CJK characters, emoji,
 * zero-width joiners, and grapheme clusters for accurate column-width
 * calculations in the TUI.
 */

// ── Internal constants ───────────────────────────────────────────────────────

const ANSI_REGEX = /\x1b\[[0-9;]*m/g;

const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, {
  granularity: 'grapheme',
});

// ── Low-level helpers ────────────────────────────────────────────────────────

/** Strip all ANSI escape sequences from a string. */
export function stripAnsi(value: string): string {
  return value.replace(ANSI_REGEX, '');
}

/** Returns true for code points that occupy zero visual columns. */
export function isZeroWidthCodePoint(code: number): boolean {
  return (
    code === 0x200d ||
    (code >= 0xfe00 && code <= 0xfe0f) ||
    (code >= 0x20d0 && code <= 0x20ff)
  );
}

/** Returns true for code points that occupy two visual columns (CJK, emoji blocks, etc.). */
export function isFullWidthCodePoint(code: number): boolean {
  return (
    code >= 0x1100 &&
    (code <= 0x115f ||
      code === 0x2329 ||
      code === 0x232a ||
      (code >= 0x2e80 && code <= 0xa4cf && code !== 0x303f) ||
      (code >= 0xac00 && code <= 0xd7a3) ||
      (code >= 0xf900 && code <= 0xfaff) ||
      (code >= 0xfe10 && code <= 0xfe19) ||
      (code >= 0xfe30 && code <= 0xfe6f) ||
      (code >= 0xff00 && code <= 0xff60) ||
      (code >= 0xffe0 && code <= 0xffe6) ||
      (code >= 0x2300 && code <= 0x23ff) ||
      (code >= 0x2600 && code <= 0x26ff) ||
      (code >= 0x2700 && code <= 0x27bf) ||
      (code >= 0x2b00 && code <= 0x2bff) ||
      (code >= 0x1f000 && code <= 0x1ffff) ||
      (code >= 0x20000 && code <= 0x3fffd))
  );
}

// ── Grapheme iteration ───────────────────────────────────────────────────────

type AnsiToken =
  | { type: 'escape'; value: string }
  | { type: 'grapheme'; value: string };

/**
 * Iterate over a string yielding ANSI escape sequences and grapheme clusters
 * as separate tokens. This allows width-aware wrapping that preserves color.
 */
export function* iterateAnsiGraphemes(text: string): Generator<AnsiToken> {
  let i = 0;
  while (i < text.length) {
    if (text[i] === '\x1b') {
      const match = text.slice(i).match(/^\x1b\[[0-9;]*m/);
      if (match) {
        yield { type: 'escape', value: match[0] };
        i += match[0].length;
        continue;
      }
    }

    const iterator = GRAPHEME_SEGMENTER.segment(text.slice(i))[Symbol.iterator]();
    const next = iterator.next();
    const segment = next.value;
    if (!segment) break;
    yield { type: 'grapheme', value: segment.segment };
    i += segment.segment.length;
  }
}

// ── Width measurement ────────────────────────────────────────────────────────

/** Calculate the visible column width of a string, ignoring ANSI escapes. */
export function visibleLength(value: string): number {
  const stripped = stripAnsi(value);
  let width = 0;
  for (const { segment } of GRAPHEME_SEGMENTER.segment(stripped)) {
    const code = segment.codePointAt(0);
    if (code === undefined || isZeroWidthCodePoint(code)) continue;
    width += isFullWidthCodePoint(code) ? 2 : 1;
  }
  return width;
}

// ── Word splitting ───────────────────────────────────────────────────────────

/**
 * Split a single "word" (which may contain ANSI sequences) into segments
 * that each fit within the given column width.
 */
export function splitAnsiWord(text: string, width: number): string[] {
  const segments: string[] = [];
  let current = '';
  let currentLength = 0;

  for (const token of iterateAnsiGraphemes(text)) {
    if (token.type === 'escape') {
      current += token.value;
      continue;
    }

    const graphemeWidth = visibleLength(token.value);
    if (currentLength + graphemeWidth > width && currentLength > 0) {
      segments.push(current);
      current = '';
      currentLength = 0;
    }

    current += token.value;
    currentLength += graphemeWidth;

    if (currentLength >= width) {
      segments.push(current);
      current = '';
      currentLength = 0;
    }
  }

  if (current) {
    segments.push(current);
  }

  return segments.length ? segments : [''];
}

// ── Line wrapping ────────────────────────────────────────────────────────────

/**
 * Wrap ANSI-colored text to fit within a given column width, preserving
 * word boundaries where possible and falling back to character-level
 * splitting for oversized words.
 */
export function wrapAnsiText(text: string, width: number): string[] {
  if (width <= 0) return [text];

  const words = text.split(/(\s+)/);
  const lines: string[] = [];
  let current = '';
  let currentLength = 0;

  for (const word of words) {
    const trimmed = word.trim();
    if (trimmed.length === 0) {
      if (currentLength + visibleLength(word) <= width) {
        current += word;
        currentLength += visibleLength(word);
      }
      continue;
    }

    const wordLength = visibleLength(word);
    const separator = currentLength > 0 ? 1 : 0;

    if (currentLength + separator + wordLength <= width) {
      if (separator) {
        current += ' ';
        currentLength += 1;
      }
      current += word;
      currentLength += wordLength;
      continue;
    }

    if (currentLength > 0) {
      lines.push(current);
      current = '';
      currentLength = 0;
    }

    if (wordLength <= width) {
      current = word;
      currentLength = wordLength;
      continue;
    }

    const pieces = splitAnsiWord(word, width);
    for (let i = 0; i < pieces.length; i += 1) {
      if (i < pieces.length - 1) {
        lines.push(pieces[i]);
      } else {
        current = pieces[i];
        currentLength = visibleLength(pieces[i]);
      }
    }
  }

  if (currentLength > 0) {
    lines.push(current);
  }

  return lines.length ? lines : [''];
}

// ── Padding ──────────────────────────────────────────────────────────────────

/** Pad a string with trailing spaces to reach a target visible width. */
export function padAnsi(value: string, width: number): string {
  const padding = width - visibleLength(value);
  return value + ' '.repeat(Math.max(0, padding));
}
