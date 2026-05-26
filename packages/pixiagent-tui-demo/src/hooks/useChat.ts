/**
 * Hook encapsulating all chat state: message history, pending status,
 * usage tracking, and backend interaction.
 */

import { useCallback, useMemo, useState } from 'react';

import type { ChatLine, UsageSnapshot } from '../types';
import { EMPTY_USAGE } from '../constants';
import { parseCommand } from '../utils/command';
import { PixiAgentBackend, type ChatResponse } from '../backend';
import type { DemoConfig } from '../env';

// ── Unique ID generator ──────────────────────────────────────────────────────

let nextMessageId = 0;

function createMessageId(): string {
  nextMessageId += 1;
  return `msg-${nextMessageId}-${Date.now()}`;
}

// ── Public types ─────────────────────────────────────────────────────────────

export type ChatState = {
  history: ChatLine[];
  pending: boolean;
  statusLabel: string;
  lastRequestUsage: UsageSnapshot;
  sessionUsage: UsageSnapshot;
};

export type ChatActions = {
  /**
   * Process text input — routes to command handler or sends as a user message.
   * Returns a promise that resolves when the action completes.
   */
  submitInput: (text: string) => Promise<void>;
  /** Interrupt the current in-flight request. */
  interrupt: (reason?: string) => void;
};

// ── Hook implementation ──────────────────────────────────────────────────────

export function useChat(
  config: DemoConfig,
  onExit: () => Promise<void>,
): ChatState & ChatActions {
  const [history, setHistory] = useState<ChatLine[]>([]);
  const [pending, setPending] = useState(false);
  const [statusLabel, setStatusLabel] = useState('Ready');
  const [lastRequestUsage, setLastRequestUsage] = useState<UsageSnapshot>(EMPTY_USAGE);
  const [sessionUsage, setSessionUsage] = useState<UsageSnapshot>(EMPTY_USAGE);

  const backend = useMemo(
    () =>
      new PixiAgentBackend(config.modelOptions, {
        modelRequestTimeout: config.modelRequestTimeout,
        maxIterations: config.maxIterations,
        maxModelRequestRetries: config.maxModelRequestRetries,
      }),
    [config],
  );

  const addMessage = useCallback(
    (role: ChatLine['role'], content: string) => {
      setHistory((prev) => [
        ...prev,
        { id: createMessageId(), role, content },
      ]);
    },
    [],
  );

  const applyUsage = useCallback((response: ChatResponse) => {
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
  }, []);

  const handleCommand = useCallback(
    async (text: string): Promise<boolean> => {
      const parsed = parseCommand(text);
      if (!parsed) return false;

      const commands: Record<string, (args: string) => Promise<void>> = {
        exit: async () => {
          await onExit();
        },
        cancel: async (args: string) => {
          if (!pending) {
            addMessage('assistant', '[cancel] No active request to interrupt.');
            setStatusLabel('Ready');
            return;
          }
          backend.interrupt(args.length > 0 ? args : undefined);
          setStatusLabel('Cancelling...');
        },
      };

      const handler = commands[parsed.name];
      if (!handler) {
        addMessage('assistant', `[command] Unknown command: /${parsed.name}`);
        setStatusLabel('Ready');
        return true;
      }

      await handler(parsed.args);
      return true;
    },
    [addMessage, backend, onExit, pending],
  );

  const submitInput = useCallback(
    async (text: string): Promise<void> => {
      const trimmed = text.trim();
      if (!trimmed) return;

      if (await handleCommand(trimmed)) return;
      if (pending) return;

      setPending(true);
      addMessage('user', trimmed);
      setStatusLabel('Thinking...');

      try {
        const response: ChatResponse = await backend.sendUserMessage(trimmed);
        applyUsage(response);
        addMessage('assistant', response.text);
        setStatusLabel('Ready');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        addMessage('assistant', `[error] ${message}`);
        setStatusLabel('Request failed');
      } finally {
        setPending(false);
      }
    },
    [addMessage, applyUsage, backend, handleCommand, pending],
  );

  const interrupt = useCallback(
    (reason?: string) => {
      backend.interrupt(reason);
    },
    [backend],
  );

  return {
    history,
    pending,
    statusLabel,
    lastRequestUsage,
    sessionUsage,
    submitInput,
    interrupt,
  };
}
