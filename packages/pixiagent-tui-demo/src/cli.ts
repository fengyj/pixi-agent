import process from 'node:process';
import {
  BoxRenderable,
  createCliRenderer,
  MarkdownRenderable,
  ScrollBoxRenderable,
  SyntaxStyle,
  TextRenderable,
  TextareaRenderable,
} from '@opentui/core';
import { Observation } from '@pixiagent/core/observation';
import { loadConfigFromEnv } from './env';
import { PixiAgentBackend, type ChatResponse } from './backend';

const { setupObservability, shutdownObservability } = Observation;

type ChatLine = {
  role: 'system' | 'user' | 'assistant';
  content: string;
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

async function main(): Promise<void> {
  const config = loadConfigFromEnv();
  if (config.observability.enabled) {
    const observabilityOptions = {
      ...config.observability.options,
    };

    // TUI owns stdout; writing structured logs to console will corrupt the screen.
    const logging = observabilityOptions.logging;
    if (logging && !('rootLogger' in logging) && !('logOptions' in logging)) {
      observabilityOptions.logging = {
        ...logging,
        outputToConsole: false,
      };
    }

    await setupObservability(observabilityOptions);
  }

  const backend = new PixiAgentBackend(config.modelOptions, {
    modelRequestTimeout: config.modelRequestTimeout,
    maxIterations: config.maxIterations,
    maxModelRequestRetries: config.maxModelRequestRetries,
  });

  const renderer = await createCliRenderer({
    exitOnCtrlC: false,
    useMouse: false,
  });
  let exiting = false;

  const exitGracefully = async (exitCode = 0): Promise<void> => {
    if (exiting) return;
    exiting = true;
    try {
      renderer.destroy();
    } catch {
      // ignore renderer destroy errors during exit
    }
    try {
      await shutdownObservability();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      process.stderr.write(`[observability shutdown error] ${message}\n`);
    } finally {
      process.exit(exitCode);
    }
  };

  const layout = new BoxRenderable(renderer, {
    id: 'layout',
    flexDirection: 'column',
    width: '100%',
    height: '100%',
    padding: 1,
    border: true,
    borderStyle: 'rounded',
    borderColor: '#3a4a5a',
  });

  const header = new TextRenderable(renderer, {
    id: 'header',
    content: `PixiAgent TUI Demo | model=${config.modelOptions.model} | base=${config.modelOptions.baseUrl}`,
    fg: '#89b4fa',
  });

  const messagePanel = new ScrollBoxRenderable(renderer, {
    id: 'message-panel',
    flexGrow: 1,
    flexShrink: 1,
    width: '100%',
    border: true,
    borderStyle: 'single',
    borderColor: '#4b5563',
    padding: 1,
    marginTop: 1,
    marginBottom: 1,
    stickyScroll: true,
    stickyStart: 'bottom',
  });

  const messageMarkdown = new MarkdownRenderable(renderer, {
    id: 'message-markdown',
    content: '',
    syntaxStyle: SyntaxStyle.create(),
    fg: '#e5e7eb',
  });

  const usageSummary = new TextRenderable(renderer, {
    id: 'usage-summary',
    content: '',
    fg: '#94a3b8',
    height: 1,
  });

  const status = new TextRenderable(renderer, {
    id: 'status',
    content: `Ready. API key from ${config.apiKeyVarName}. Ctrl+Enter or Ctrl+S to send, Enter for newline, /exit to quit.`,
    fg: '#f9e2af',
    height: 1,
  });

  const input = new TextareaRenderable(renderer, {
    id: 'chat-input',
    width: '100%',
    height: 4,
    placeholder: 'Type your message... (Ctrl+Enter or Ctrl+S to send)',
    backgroundColor: '#111827',
    focusedBackgroundColor: '#1f2937',
    textColor: '#f9fafb',
    cursorColor: '#93c5fd',
    wrapMode: 'word',
    keyBindings: [
      { name: 'return', ctrl: true, action: 'submit' },
      { name: 'enter', ctrl: true, action: 'submit' },
      { name: 'return', meta: true, action: 'submit' },
      { name: 'enter', meta: true, action: 'submit' },
      { name: 's', ctrl: true, action: 'submit' },
    ],
  });

  messagePanel.content.add(messageMarkdown);

  const footer = new BoxRenderable(renderer, {
    id: 'footer',
    flexDirection: 'column',
    width: '100%',
    height: 2,
    flexShrink: 0,
    border: false,
  });
  footer.add(usageSummary);
  footer.add(status);

  layout.add(header);
  layout.add(messagePanel);
  layout.add(footer);
  layout.add(input);
  renderer.root.add(layout);

  const history: ChatLine[] = [];
  let termWidth = process.stdout.columns ?? 120;
  let termHeight = process.stdout.rows ?? 40;
  let pending = false;
  let statusLabel = 'Ready';
  let lastRequestUsage = createEmptyUsage();
  let sessionUsage = createEmptyUsage();

  function createEmptyUsage(): { inputTokens: number; outputTokens: number; totalTokens: number; cacheReadTokens: number } {
    return {
      inputTokens: 0,
      outputTokens: 0,
      totalTokens: 0,
      cacheReadTokens: 0,
    };
  }

  const formatCompactToken = (value: number): string => {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toFixed(2).replace(/\.?(0+)$/, '')}m`.padStart(6);
    }
    if (value >= 100_000) {
      return `${Math.round(value / 1_000)}k`.padStart(6);
    }
    if (value >= 10_000) {
      return `${(value / 1_000).toFixed(1).replace(/\.0$/, '')}k`.padStart(6);
    }
    return value.toLocaleString('en-US').padStart(6);
  };

  const formatUsageSummary = (label: string, usage: { inputTokens: number; outputTokens: number; cacheReadTokens: number; totalTokens: number }): string =>
    `${label} ${formatCompactToken(usage.inputTokens)} | ${formatCompactToken(usage.outputTokens)} | ${formatCompactToken(usage.cacheReadTokens)} | ${formatCompactToken(usage.totalTokens)}`;

  const maxScrollTop = (): number => Math.max(0, messagePanel.scrollHeight - messagePanel.viewport.height);

  const pushHistory = (line: ChatLine): void => {
    history.push(line);
  };

  const lineWidth = (): number => Math.max(0, termWidth - 4);

  const rightAlign = (text: string): string => {
    const width = lineWidth();
    if (text.length >= width) return text;
    return text.padStart(width);
  };

  const formatSessionLine = (): string => rightAlign(formatUsageSummary('Total (I|O|C|T):', sessionUsage));

  const formatStatusLine = (): string => {
    const maxTop = maxScrollTop();
    const scrollInfo =
      messagePanel.scrollTop >= maxTop
        ? ' | scroll: bottom'
        : messagePanel.scrollTop <= 0
        ? ' | scroll: top'
        : ' | scroll: active';
    const base = `${statusLabel}${scrollInfo}`;
    const requestInfo = formatUsageSummary('Last  (I|O|C|T):', lastRequestUsage);
    const width = lineWidth();
    if (base.length + requestInfo.length + 1 <= width) {
      return `${base}${requestInfo.padStart(width - base.length)}`;
    }
    return `${base} ${requestInfo}`;
  };

  const setStatus = (next: string): void => {
    statusLabel = next;
    status.content = formatStatusLine();
    usageSummary.content = formatSessionLine();
  };

  const renderHistory = (): void => {
    const dividerWidth = Math.max(24, (messagePanel.viewport.width || Math.max(24, termWidth - 8)) - 1);
    messageMarkdown.content = renderMessageContent(history, dividerWidth);
    setStatus(statusLabel);
  };

  renderHistory();
  input.focus();

  renderer.keyInput.on('keypress', (key) => {
    let changed = false;

    if (key.ctrl && key.name === 'c') {
      void exitGracefully(0);
      return;
    }

    if (
      (key.ctrl && (key.name === 'return' || key.name === 'enter')) ||
      (key.meta && (key.name === 'return' || key.name === 'enter')) ||
      (key.ctrl && key.name === 's')
    ) {
      try {
        input.submit();
      } catch {
        // ignore if submit is unavailable
      }
      return;
    }

    const pageDelta = Math.max(3, Math.floor(messagePanel.viewport.height / 2));
    const maxTop = maxScrollTop();

    if (key.name === 'pageup' || (key.ctrl && key.name === 'up')) {
      messagePanel.scrollTop = Math.max(0, messagePanel.scrollTop - pageDelta);
      changed = true;
    } else if (key.name === 'pagedown' || (key.ctrl && key.name === 'down')) {
      messagePanel.scrollTop = Math.min(maxTop, messagePanel.scrollTop + pageDelta);
      changed = true;
    } else if (key.ctrl && key.name === 'u') {
      messagePanel.scrollTop = Math.max(0, messagePanel.scrollTop - 1);
      changed = true;
    } else if (key.ctrl && key.name === 'd') {
      messagePanel.scrollTop = Math.min(maxTop, messagePanel.scrollTop + 1);
      changed = true;
    } else if (key.ctrl && key.name === 'home') {
      messagePanel.scrollTop = 0;
      changed = true;
    } else if (key.ctrl && key.name === 'end') {
      messagePanel.scrollTop = maxTop;
      changed = true;
    }

    if (changed) {
      setStatus(statusLabel);
    }
  });

  renderer.on('resize', (width, height) => {
    termWidth = width;
    termHeight = height;
    renderHistory();
  });

  input.onSubmit = async () => {
    const rawValue = input.plainText;
    const value = rawValue.trim();
    if (!value || pending) {
      return;
    }

    if (value === '/exit') {
      await exitGracefully(0);
      return;
    }

    pending = true;
    input.selectAll();
    input.deleteSelection();
    pushHistory({ role: 'user', content: value });
    setStatus('assistant is thinking...');
    renderHistory();

    try {
      const response: ChatResponse = await backend.sendUserMessage(value);
      lastRequestUsage = {
        inputTokens: response.usage.inputTokens,
        outputTokens: response.usage.outputTokens,
        totalTokens: response.usage.totalTokens,
        cacheReadTokens: response.usage.cacheReadTokens ?? 0,
      };
      sessionUsage = {
        inputTokens: response.sessionUsage.inputTokens,
        outputTokens: response.sessionUsage.outputTokens,
        totalTokens: response.sessionUsage.totalTokens,
        cacheReadTokens: response.sessionUsage.cacheReadTokens ?? 0,
      };
      pushHistory({ role: 'assistant', content: response.text });
      setStatus('Ready');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      pushHistory({ role: 'assistant', content: `[error] ${message}` });
      setStatus('Request failed');
    } finally {
      pending = false;
      renderHistory();
      input.focus();
    }
  };
}

main().catch(async (error) => {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Fatal error: ${message}\n`);
  try {
    await shutdownObservability();
  } catch {
    // ignore shutdown errors in fatal path
  }
  process.exitCode = 1;
});
