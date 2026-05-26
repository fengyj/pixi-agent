/**
 * Root application component — orchestrates child components, hooks, and
 * keyboard input dispatch.
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Box, Text, useApp, useInput, useStdout } from 'ink';

import type { DemoConfig } from '../env';
import { COLORS, DEFAULT_COLUMNS, DEFAULT_ROWS, CHROME_ROWS, MIN_CONTENT_WIDTH } from '../constants';
import { Observation } from '@pixiagent/core';
import { renderMessageContent } from '../markdown/renderer';
import { useChat } from '../hooks/useChat';
import { useScrollableView } from '../hooks/useScrollableView';

import { MessagePanel } from './MessagePanel';
import { InputPanel } from './InputPanel';
import { StatusBar } from './StatusBar';

const { setupObservability, shutdownObservability } = Observation;

type AppProps = {
  config: DemoConfig;
};

export const App = ({ config }: AppProps): JSX.Element => {
  const [inputText, setInputText] = useState('');
  const [shuttingDown, setShuttingDown] = useState(false);

  const { exit } = useApp();
  const { stdout } = useStdout();
  const columns = stdout.columns ?? DEFAULT_COLUMNS;
  const rows = stdout.rows ?? DEFAULT_ROWS;

  // ── Graceful shutdown ────────────────────────────────────────────────────

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

  // ── Chat state ───────────────────────────────────────────────────────────

  const chat = useChat(config, exitGracefully);

  // ── Observability lifecycle ──────────────────────────────────────────────

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
  }, [config.observability]);

  // ── Message rendering ────────────────────────────────────────────────────

  const markdownWidth = Math.max(MIN_CONTENT_WIDTH, columns - 10);
  const dividerWidth = Math.max(MIN_CONTENT_WIDTH, columns - 8);

  const messageContent = useMemo(
    () => renderMessageContent(chat.history, dividerWidth, markdownWidth),
    [chat.history, dividerWidth, markdownWidth],
  );

  const messageLines = useMemo(
    () => messageContent.split('\n'),
    [messageContent],
  );

  // ── Scrollable viewport ──────────────────────────────────────────────────

  const messagePanelHeight = Math.max(4, rows - CHROME_ROWS);
  const scroll = useScrollableView(messageLines, messagePanelHeight);

  // Reset scroll to bottom when new messages arrive.
  useEffect(() => {
    scroll.actions.resetScroll();
    // Intentionally depend only on history length to avoid excessive resets.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat.history.length]);

  // ── Input submission ─────────────────────────────────────────────────────

  const submitInput = useCallback(async () => {
    const text = inputText;
    if (!text.trim()) return;
    setInputText('');
    await chat.submitInput(text);
  }, [chat, inputText]);

  // ── Keyboard dispatch ────────────────────────────────────────────────────

  useInput((input, key) => {
    // Global: Ctrl+C always exits.
    if (key.ctrl && input === 'c') {
      void exitGracefully();
      return;
    }

    // Submit: Ctrl+S.
    if (key.ctrl && input === 's') {
      void submitInput();
      return;
    }

    // Newline: Enter.
    if (key.return || key.enter) {
      setInputText((prev) => `${prev}\n`);
      return;
    }

    // Backspace.
    if (key.backspace) {
      setInputText((prev) => prev.slice(0, -1));
      return;
    }

    // Scroll navigation.
    const pageUp = key.pageUp || key.pageup;
    const pageDown = key.pageDown || key.pagedown;
    const arrowUp = key.up || key.upArrow;
    const arrowDown = key.down || key.downArrow;

    if (pageUp || (key.ctrl && arrowUp)) {
      scroll.actions.pageUp();
      return;
    }
    if (pageDown || (key.ctrl && arrowDown)) {
      scroll.actions.pageDown();
      return;
    }
    if (key.ctrl && key.u) {
      scroll.actions.lineUp();
      return;
    }
    if (key.ctrl && key.d) {
      scroll.actions.lineDown();
      return;
    }
    if (key.ctrl && key.home) {
      scroll.actions.scrollToTop();
      return;
    }
    if (key.ctrl && key.end) {
      scroll.actions.scrollToBottom();
      return;
    }

    // Character input.
    if (!key.ctrl && !key.meta && input) {
      setInputText((prev) => prev + input);
    }
  });

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <Box
      flexDirection="column"
      paddingTop={1}
      paddingRight={1}
      paddingBottom={0}
      paddingLeft={1}
      borderStyle="round"
      borderColor={COLORS.frame}
    >
      <Text color={COLORS.primary}>
        PixiAgent TUI Demo | model={config.modelOptions.model} | base=
        {config.modelOptions.baseUrl}
      </Text>

      <MessagePanel
        visibleLines={scroll.visibleLines}
        height={messagePanelHeight}
      />

      <StatusBar
        statusLabel={chat.statusLabel}
        scrollPositionText={scroll.scrollPositionText}
        lastRequestUsage={chat.lastRequestUsage}
        sessionUsage={chat.sessionUsage}
      />

      <InputPanel text={inputText} />
    </Box>
  );
};
