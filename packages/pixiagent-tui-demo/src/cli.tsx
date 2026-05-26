
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Box, Text, render, useApp, useInput, useStdout } from 'ink';
import * as marked from 'marked';
import { markedTerminal } from 'marked-terminal';
import { loadConfigFromEnv } from './env';
import { Observation } from '@pixiagent/core';
const { setupObservability, shutdownObservability } = Observation;
import { PixiAgentBackend, type ChatResponse } from './backend';

type ChatLine = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

type ParsedCommand = {
  name: string;
  args: string;
};

const DISPLAY_ROLE_LABELS = {
  user: 'USER',
  assistant: 'AGENT',
} as const;

const ROLE_LABEL_WIDTH = Math.max(
  DISPLAY_ROLE_LABELS.user.length,
  DISPLAY_ROLE_LABELS.assistant.length,
);

function createRoleDivider(label: string, width: number): string {
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

function parseCommand(input: string): ParsedCommand | null {
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

const ANSI_REGEX = /\x1b\[[0-9;]*m/g;
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

function stripAnsi(value: string): string {
  return value.replace(ANSI_REGEX, '');
}

function isZeroWidthCodePoint(code: number): boolean {
  return (
    code === 0x200d ||
    (code >= 0xfe00 && code <= 0xfe0f) ||
    (code >= 0x20d0 && code <= 0x20ff)
  );
}

function isFullWidthCodePoint(code: number): boolean {
  return (
    code >= 0x1100 &&
    (
      code <= 0x115f ||
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
      (code >= 0x20000 && code <= 0x3fffd)
    )
  );
}

function* iterateAnsiGraphemes(text: string) {
  let i = 0;
  while (i < text.length) {
    if (text[i] === '\x1b') {
      const match = text.slice(i).match(/^\x1b\[[0-9;]*m/);
      if (match) {
        yield { type: 'escape' as const, value: match[0] };
        i += match[0].length;
        continue;
      }
    }

    const iterator = GRAPHEME_SEGMENTER.segment(text.slice(i))[Symbol.iterator]();
    const next = iterator.next();
    const segment = next.value;
    if (!segment) break;
    yield { type: 'grapheme' as const, value: segment.segment };
    i += segment.segment.length;
  }
}

function visibleLength(value: string): number {
  const stripped = stripAnsi(value);
  let width = 0;
  for (const { segment } of GRAPHEME_SEGMENTER.segment(stripped)) {
    const code = segment.codePointAt(0);
    if (code === undefined || isZeroWidthCodePoint(code)) continue;
    width += isFullWidthCodePoint(code) ? 2 : 1;
  }
  return width;
}

function splitAnsiWord(text: string, width: number): string[] {
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

function wrapAnsiText(text: string, width: number): string[] {
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

function padAnsi(value: string, width: number): string {
  const padding = width - visibleLength(value);
  return value + ' '.repeat(Math.max(0, padding));
}

function renderTableCell(cell: unknown, renderer: marked.Renderer): string {
  if (typeof cell === 'object' && cell !== null) {
    // @ts-expect-error marked parser types
    return renderer.parser.parseInline((cell as any).tokens);
  }
  return String(cell ?? '');
}

function renderAsciiTable(token: any, width: number, renderer: marked.Renderer): string {
  const headerCells = token.header.map((cell: unknown) => renderTableCell(cell, renderer));
  const bodyRows = token.rows.map((row: unknown[]) => row.map((cell) => renderTableCell(cell, renderer)));
  const cols = Math.max(headerCells.length, ...bodyRows.map((row) => row.length));

  const allRows = [headerCells, ...bodyRows];
  const desiredWidths = new Array(cols).fill(0);
  allRows.forEach((row) => {
    row.forEach((cell, index) => {
      desiredWidths[index] = Math.max(desiredWidths[index] ?? 0, visibleLength(cell));
    });
  });

  const totalPadding = cols * 2;
  const totalSeparators = cols + 1;
  const available = Math.max(10, width - totalPadding - totalSeparators);
  const totalDesired = desiredWidths.reduce((sum, w) => sum + w, 0) || cols * 10;

  let colWidths = desiredWidths.map((desired) => Math.max(4, Math.min(desired, Math.floor((desired * available) / totalDesired))));

  let allocated = colWidths.reduce((sum, v) => sum + v, 0);
  let remaining = available - allocated;

  // If there is no space left due to rounding, shrink the widest columns first.
  while (allocated > available) {
    const largestIndex = colWidths.reduce(
      (best, width, index) => (width > colWidths[best] ? index : best),
      0,
    );
    if (colWidths[largestIndex] <= 4) break;
    colWidths[largestIndex] -= 1;
    allocated -= 1;
  }

  while (remaining > 0) {
    const smallestIndex = colWidths.reduce(
      (best, width, index) => (width < colWidths[best] ? index : best),
      0,
    );
    colWidths[smallestIndex] += 1;
    remaining -= 1;
  }

  const wrapRow = (row: string[]) => {
    const cells = new Array(cols).fill('');
    row.forEach((cell, index) => {
      cells[index] = cell;
    });
    return cells.map((cell, index) => wrapAnsiText(cell, colWidths[index]));
  };

  const headerWrapped = wrapRow(headerCells);
  const bodyWrapped = bodyRows.map(wrapRow);

  const rowHeights = [headerWrapped, ...bodyWrapped].map((row) => Math.max(...row.map((lines) => lines.length)));

  const drawBorder = (left: string, mid: string, right: string) =>
    left + colWidths.map((w) => '─'.repeat(w + 2)).join(mid) + right;

  const lines: string[] = [];
  lines.push(drawBorder('┌', '┬', '┐'));

  const renderRow = (row: string[][], height: number) => {
    for (let lineIndex = 0; lineIndex < height; lineIndex += 1) {
      const cells = row.map((cellLines, colIndex) => padAnsi(cellLines[lineIndex] ?? '', colWidths[colIndex]));
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

function renderMarkdown(content: string, width: number): string {
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
          const tokens = isTask && item.tokens?.length && item.tokens[0].type === 'checkbox' ? item.tokens.slice(1) : item.tokens;
          const content = tokens ? this.parser.parse(tokens, !!item.loose) : '';
          return '\n* ' + checkbox + content;
        }

        return '\n* ' + text;
      },
    },
  });

  return (markdown.parse(content) as string).trimEnd();
}

function renderMessageContent(lines: ChatLine[], dividerWidth: number, markdownWidth: number): string {
  const visibleLines = lines.filter((line) => line.role !== 'system');

  if (visibleLines.length === 0) {
    return 'No history yet. Send a message to begin the conversation.';
  }

  return visibleLines
    .map((line) => {
      const label = line.role === 'user' ? DISPLAY_ROLE_LABELS.user : DISPLAY_ROLE_LABELS.assistant;
      return `${createRoleDivider(label, dividerWidth)}\n${renderMarkdown(line.content, markdownWidth)}`;
    })
    .join('\n\n');
}

function createEmptyUsage() {
  return {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    cacheReadTokens: 0,
  };
}

type UsageState = ReturnType<typeof createEmptyUsage>;

const formatCompactToken = (value: number): string => {
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
};

const formatUsageTokens = (
  usage: UsageState,
): string => `${formatCompactToken(usage.inputTokens)} | ${formatCompactToken(usage.outputTokens)} | ${formatCompactToken(usage.cacheReadTokens)} | ${formatCompactToken(usage.totalTokens)}`;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

const config = loadConfigFromEnv();

const App = (): JSX.Element => {
  const [history, setHistory] = useState<ChatLine[]>([]);
  const [inputText, setInputText] = useState('');
  const [pending, setPending] = useState(false);
  const [statusLabel, setStatusLabel] = useState('Ready');
  const [lastRequestUsage, setLastRequestUsage] = useState<UsageState>(createEmptyUsage());
  const [sessionUsage, setSessionUsage] = useState<UsageState>(createEmptyUsage());
  const [scrollOffset, setScrollOffset] = useState(0);
  const [shuttingDown, setShuttingDown] = useState(false);
  const [cursorVisible, setCursorVisible] = useState(true);

  const { exit } = useApp();
  const { stdout } = useStdout();
  const stdoutColumns = stdout.columns ?? 120;
  const stdoutRows = stdout.rows ?? 40;

  const backend = useMemo(
    () =>
      new PixiAgentBackend(config.modelOptions, {
        modelRequestTimeout: config.modelRequestTimeout,
        maxIterations: config.maxIterations,
        maxModelRequestRetries: config.maxModelRequestRetries,
      }),
    [],
  );

  useEffect(() => {
    if (config.observability.enabled) {
      void setupObservability(config.observability.options).catch(() => {
        // ignore observability startup failures for demo
      });
    }

    return () => {
      void shutdownObservability().catch(() => {
        // ignore shutdown errors during unmount
      });
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCursorVisible((current) => !current);
    }, 530);
    return () => clearInterval(interval);
  }, []);

  const addHistory = useCallback((line: ChatLine) => {
    setHistory((prev) => [...prev, line]);
    setScrollOffset(0);
  }, []);

  const exitGracefully = useCallback(async () => {
    if (shuttingDown) return;
    setShuttingDown(true);
    try {
      await shutdownObservability();
    } catch {
      // ignore shutdown errors
    }
    exit();
  }, [exit, shuttingDown]);

  const handleCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const parsed = parseCommand(text);
      if (!parsed) {
        return false;
      }

      const commands: Record<string, (args: string) => Promise<void>> = {
        exit: async () => {
          await exitGracefully();
        },
        cancel: async (args: string) => {
          if (!pending) {
            addHistory({ role: 'assistant', content: '[cancel] No active request to interrupt.' });
            setStatusLabel('Ready');
            return;
          }
          backend.interrupt(args.length > 0 ? args : undefined);
          setStatusLabel('Cancelling...');
        },
      };

      const handler = commands[parsed.name];
      if (!handler) {
        addHistory({ role: 'assistant', content: `[command] Unknown command: /${parsed.name}` });
        setStatusLabel('Ready');
        return true;
      }

      await handler(parsed.args);
      return true;
    },
    [addHistory, backend, exitGracefully, pending],
  );

  const submitInput = useCallback(async () => {
    const trimmed = inputText.trim();
    if (!trimmed) return;

    setInputText('');

    if (await handleCommand(trimmed)) {
      return;
    }

    if (pending) return;

    setPending(true);
    addHistory({ role: 'user', content: trimmed });
    setStatusLabel('Thinking...');

    try {
      const response: ChatResponse = await backend.sendUserMessage(trimmed);
      setLastRequestUsage({
        inputTokens: response.usage.inputTokens,
        outputTokens: response.usage.outputTokens,
        totalTokens: response.usage.totalTokens,
        cacheReadTokens: response.usage.cacheReadTokens ?? 0,
      });
      setSessionUsage({
        inputTokens: response.sessionUsage.inputTokens,
        outputTokens: response.sessionUsage.outputTokens,
        totalTokens: response.sessionUsage.totalTokens,
        cacheReadTokens: response.sessionUsage.cacheReadTokens ?? 0,
      });
      addHistory({ role: 'assistant', content: response.text });
      setStatusLabel('Ready');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      addHistory({ role: 'assistant', content: `[error] ${message}` });
      setStatusLabel('Request failed');
    } finally {
      setPending(false);
    }
  }, [addHistory, backend, handleCommand, inputText, pending]);

  const markdownWidth = Math.max(24, stdoutColumns - 10);
  const messageContent = useMemo(
    () => renderMessageContent(history, Math.max(24, stdoutColumns - 8), markdownWidth),
    [history, markdownWidth],
  );

  const messageLines = useMemo(() => messageContent.split('\n'), [messageContent]);
  const messagePanelHeight = Math.max(4, stdoutRows - 12);
  const maxOffset = Math.max(0, messageLines.length - messagePanelHeight);
  const offset = clamp(scrollOffset, 0, maxOffset);
  const visibleLines = messageLines.slice(
    Math.max(0, messageLines.length - messagePanelHeight - offset),
    messageLines.length - offset,
  );
  const firstVisibleLine = Math.max(1, messageLines.length - messagePanelHeight - offset + 1);
  const lastVisibleLine = Math.max(0, messageLines.length - offset);
  const scrollPositionText =
    offset === 0
      ? 'bottom'
      : `${firstVisibleLine}-${lastVisibleLine}/${messageLines.length}`;

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      void exitGracefully();
      return;
    }

    if (key.ctrl && input === 's') {
      void submitInput();
      return;
    }

    if (key.return || key.enter) {
      setInputText((prev) => `${prev}\n`);
      return;
    }

    if (key.backspace) {
      setInputText((prev) => prev.slice(0, -1));
      return;
    }

    const pageUp = key.pageUp || key.pageup;
    const pageDown = key.pageDown || key.pagedown;
    const arrowUp = key.up || key.upArrow;
    const arrowDown = key.down || key.downArrow;

    if (pageUp || (key.ctrl && arrowUp)) {
      setScrollOffset((current) => clamp(current + Math.max(3, Math.floor(messagePanelHeight / 2)), 0, maxOffset));
      return;
    }

    if (pageDown || (key.ctrl && arrowDown)) {
      setScrollOffset((current) => clamp(current - Math.max(3, Math.floor(messagePanelHeight / 2)), 0, maxOffset));
      return;
    }

    if (key.ctrl && key.u) {
      setScrollOffset((current) => clamp(current + 1, 0, maxOffset));
      return;
    }

    if (key.ctrl && key.d) {
      setScrollOffset((current) => clamp(current - 1, 0, maxOffset));
      return;
    }

    if (key.ctrl && key.home) {
      setScrollOffset(maxOffset);
      return;
    }

    if (key.ctrl && key.end) {
      setScrollOffset(0);
      return;
    }

    if (!key.ctrl && !key.meta && input) {
      setInputText((prev) => prev + input);
    }
  });

  const placeholderText = 'Type your message... (Ctrl+S to send)';
  const isPlaceholder = inputText.length === 0;
  const rawInputLines = isPlaceholder ? [placeholderText] : inputText.split('\n');
  const lastLineIndex = rawInputLines.length - 1;
  const statusText = `${statusLabel} | Pos: ${scrollPositionText}`;
  const sessionUsageText = formatUsageTokens(sessionUsage);
  const requestUsageText = formatUsageTokens(lastRequestUsage);
  const cursorChar = cursorVisible ? '▉' : ' ';

  return (
    <Box flexDirection="column" paddingTop={1} paddingRight={1} paddingBottom={0} paddingLeft={1} borderStyle="round" borderColor="#3a4a5a">
      <Text color="#89b4fa">PixiAgent TUI Demo | model={config.modelOptions.model} | base={config.modelOptions.baseUrl}</Text>

      <Box flexDirection="column" borderStyle="single" borderColor="#4b5563" paddingLeft={1} paddingRight={1} paddingTop={0} paddingBottom={0} flexGrow={1} minHeight={messagePanelHeight}>
        {visibleLines.map((line, index) => (
          <Text key={`message-${index}`} wrap="wrap">
            {line}
          </Text>
        ))}
      </Box>

      <Box justifyContent="flex-end" marginTop={1}>
        <Text color="#94a3b8">Total (I|O|C|T): {sessionUsageText}</Text>
      </Box>
      <Box justifyContent="space-between">
        <Text color="#f9e2af">{statusText}</Text>
        <Text color="#f9e2af">Last  (I|O|C|T): {requestUsageText}</Text>
      </Box>

      <Box flexDirection="column" borderStyle="single" borderColor="#4b5563" paddingLeft={1} paddingRight={1} paddingTop={0} paddingBottom={0} minHeight={3} marginTop={0}>
        {rawInputLines.map((line, index) => {
          const isLast = index === lastLineIndex;
          if (isPlaceholder) {
            return (
              <Text key={`input-placeholder-${index}`} color="#6b7280">
                {cursorChar}
                {line}
              </Text>
            );
          }

          return (
            <Text key={`input-${index}`}>
              {line}
              {isLast ? cursorChar : ''}
            </Text>
          );
        })}
      </Box>
    </Box>
  );
};

render(<App />);
