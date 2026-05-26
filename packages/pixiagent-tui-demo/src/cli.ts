
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Box, Text, render, useApp, useInput, useStdoutDimensions } from 'ink';
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

function renderMessageContent(lines: ChatLine[], dividerWidth: number): string {
  const visibleLines = lines.filter((line) => line.role !== 'system');

  return visibleLines
    .map((line) => {
      const label =
        line.role === 'user'
          ? DISPLAY_ROLE_LABELS.user
          : DISPLAY_ROLE_LABELS.assistant;
      return `${createRoleDivider(label, dividerWidth)}\n${line.content}`;
    })
    .join('\n\n');
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

const formatUsageSummary = (
  label: string,
  usage: UsageState,
): string => `${label} ${formatCompactToken(usage.inputTokens)} | ${formatCompactToken(usage.outputTokens)} | ${formatCompactToken(usage.cacheReadTokens)} | ${formatCompactToken(usage.totalTokens)}`;

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

  const { exit } = useApp();
  const { stdoutColumns = 120, stdoutRows = 40 } = useStdoutDimensions();

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
    setStatusLabel('assistant is thinking...');

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

  const messageContent = useMemo(
    () => renderMessageContent(history, Math.max(24, stdoutColumns - 8)),
    [history, stdoutColumns],
  );

  const messageLines = useMemo(() => messageContent.split('\n'), [messageContent]);
  const messagePanelHeight = Math.max(4, stdoutRows - 12);
  const maxOffset = Math.max(0, messageLines.length - messagePanelHeight);
  const offset = clamp(scrollOffset, 0, maxOffset);
  const visibleLines = messageLines.slice(
    Math.max(0, messageLines.length - messagePanelHeight - offset),
    messageLines.length - offset,
  );

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      void exitGracefully();
      return;
    }

    if ((key.ctrl || key.meta) && (key.return || key.enter)) {
      void submitInput();
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

    if (key.pageup || (key.ctrl && key.up)) {
      setScrollOffset((current) => clamp(current + Math.max(3, Math.floor(messagePanelHeight / 2)), 0, maxOffset));
      return;
    }

    if (key.pagedown || (key.ctrl && key.down)) {
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

  const inputLines = inputText.length > 0 ? inputText.split('\n') : [''];
  const statusText = `${statusLabel}${offset === 0 ? ' | scroll: bottom' : ` | scroll: older (${offset})`}`;
  const usageText = formatUsageSummary('Total (I|O|C|T):', sessionUsage);

  return (
    <Box flexDirection="column" padding={1} borderStyle="round" borderColor="#3a4a5a">
      <Text color="#89b4fa">PixiAgent TUI Demo | model={config.modelOptions.model} | base={config.modelOptions.baseUrl}</Text>

      <Box flexDirection="column" borderStyle="single" borderColor="#4b5563" padding={1} flexGrow={1} minHeight={messagePanelHeight}>
        {visibleLines.map((line, index) => (
          <Text key={index}>{line}</Text>
        ))}
      </Box>

      <Text color="#94a3b8">{usageText}</Text>
      <Text color="#f9e2af">{statusText}</Text>

      <Box flexDirection="column" borderStyle="single" borderColor="#4b5563" padding={1} minHeight={4} marginTop={1}>
        {inputLines.map((line, index) => (
          <Text key={index}>{line || ' '}</Text>
        ))}
      </Box>

      <Text color="#94a3b8" wrap="truncate-end">
        Controls: Ctrl+Enter or Ctrl+S to send · Enter for newline · /exit to quit · /cancel to interrupt
      </Text>
    </Box>
  );
};

render(<App />);
