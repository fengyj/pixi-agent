import process from 'node:process';
import {
  BoxRenderable,
  createCliRenderer,
  TextRenderable,
  TextareaRenderable,
} from '@opentui/core';
import { Observation } from '@pixiagent/core/observation';
import { loadConfigFromEnv } from './env';
import { PixiAgentBackend } from './backend';

const { setupObservability, shutdownObservability } = Observation;

type ChatLine = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

function wrapLine(line: string, width: number): string[] {
  if (line.length <= width) {
    return [line];
  }

  const wrapped: string[] = [];
  let rest = line;
  while (rest.length > width) {
    wrapped.push(rest.slice(0, width));
    rest = rest.slice(width);
  }
  if (rest.length > 0) {
    wrapped.push(rest);
  }
  return wrapped;
}

function toDisplayLines(lines: ChatLine[], width: number): string[] {
  const output: string[] = [];

  for (const line of lines) {
    const prefix = line.role === 'system' ? '[system] ' : line.role === 'user' ? 'you> ' : 'assistant> ';
    const rawLines = line.content.split('\n');

    for (let i = 0; i < rawLines.length; i += 1) {
      const withPrefix = i === 0 ? `${prefix}${rawLines[i]}` : `${' '.repeat(prefix.length)}${rawLines[i]}`;
      output.push(...wrapLine(withPrefix, width));
    }

    output.push('');
  }

  if (output.length > 0 && output[output.length - 1] === '') {
    output.pop();
  }

  return output;
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

  const messagePanel = new BoxRenderable(renderer, {
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
  });

  const messageText = new TextRenderable(renderer, {
    id: 'message-text',
    content: '',
    fg: '#e5e7eb',
  });

  const status = new TextRenderable(renderer, {
    id: 'status',
    content: `Ready. API key from ${config.apiKeyVarName}. Ctrl+Enter or Ctrl+S to send, Enter for newline, /exit to quit.`,
    fg: '#f9e2af',
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

  messagePanel.add(messageText);
  layout.add(header);
  layout.add(messagePanel);
  layout.add(status);
  layout.add(input);
  renderer.root.add(layout);

  const history: ChatLine[] = [
    {
      role: 'system',
      content: 'Welcome.',
    },
  ];
  let termWidth = process.stdout.columns ?? 120;
  let termHeight = process.stdout.rows ?? 40;
  let scrollLineOffset = 0;
  let pending = false;
  let statusLabel = 'Ready';

  const contentWidth = (): number => Math.max(24, termWidth - 8);

  const viewportLineCount = (): number => Math.max(6, termHeight - 12);

  const allDisplayLines = (): string[] => toDisplayLines(history, contentWidth());

  const maxScrollOffset = (): number => Math.max(0, allDisplayLines().length - viewportLineCount());

  const clampScrollOffset = (): void => {
    const maxOffset = maxScrollOffset();
    if (scrollLineOffset < 0) scrollLineOffset = 0;
    if (scrollLineOffset > maxOffset) scrollLineOffset = maxOffset;
  };

  const pushHistory = (line: ChatLine): void => {
    const before = allDisplayLines().length;
    history.push(line);
    const after = allDisplayLines().length;
    if (scrollLineOffset > 0) {
      // Keep the same viewport anchored when user is browsing older messages.
      scrollLineOffset += after - before;
    }
    clampScrollOffset();
  };

  const setStatus = (next: string): void => {
    statusLabel = next;
    const maxOffset = maxScrollOffset();
    const scrollInfo =
      scrollLineOffset > 0
        ? ` | scroll(lines): +${scrollLineOffset}/${maxOffset}`
        : ' | scroll(lines): bottom';
    status.content = `${statusLabel}${scrollInfo}`;
  };

  const renderHistory = (): void => {
    clampScrollOffset();
    const allLines = allDisplayLines();
    const endExclusive = Math.max(0, allLines.length - scrollLineOffset);
    const start = Math.max(0, endExclusive - viewportLineCount());
    const visibleLines = allLines.slice(start, endExclusive);
    messageText.content = visibleLines.join('\n');
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

    if (key.name === 'pageup' || (key.ctrl && key.name === 'up')) {
      scrollLineOffset += Math.max(3, Math.floor(viewportLineCount() / 2));
      changed = true;
    } else if (key.name === 'pagedown' || (key.ctrl && key.name === 'down')) {
      scrollLineOffset -= Math.max(3, Math.floor(viewportLineCount() / 2));
      changed = true;
    } else if (key.ctrl && key.name === 'u') {
      scrollLineOffset += 1;
      changed = true;
    } else if (key.ctrl && key.name === 'd') {
      scrollLineOffset -= 1;
      changed = true;
    } else if (key.ctrl && key.name === 'home') {
      scrollLineOffset = maxScrollOffset();
      changed = true;
    } else if (key.ctrl && key.name === 'end') {
      scrollLineOffset = 0;
      changed = true;
    }

    if (changed) {
      renderHistory();
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
      const answer = await backend.sendUserMessage(value);
      pushHistory({ role: 'assistant', content: answer });
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
