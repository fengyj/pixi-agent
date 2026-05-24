/**
 * Minimal TUI smoke-test — no observability, no agent, no env overrides.
 * Run with:  bun run src/minimal.ts
 *
 * If garbled escape sequences still appear here, the root cause is OpenTUI
 * itself (or the WSL2/Windows Terminal environment), not our application code.
 */
import {
  BoxRenderable,
  createCliRenderer,
  TextRenderable,
  TextareaRenderable,
} from '@opentui/core';

async function main(): Promise<void> {
  const renderer = await createCliRenderer({
    exitOnCtrlC: false,
    targetFps: 30,
  });

  const layout = new BoxRenderable(renderer, {
    flexDirection: 'column',
    width: '100%',
    height: '100%',
    border: true,
    borderStyle: 'rounded',
  });

  const display = new TextRenderable(renderer, {
    content: 'Minimal TUI — type a message and press Ctrl+Enter to echo it.',
    flexGrow: 1,
    fg: '#e5e7eb',
  });

  const input = new TextareaRenderable(renderer, {
    width: '100%',
    height: 4,
    placeholder: 'Type here… (Ctrl+Enter to echo, Ctrl+C to quit)',
    keyBindings: [{ name: 'return', ctrl: true, action: 'submit' }],
  });

  layout.add(display);
  layout.add(input);
  renderer.root.add(layout);

  input.focus();

  renderer.keyInput.on('keypress', (key) => {
    if (key.ctrl && key.name === 'c') {
      renderer.destroy();
      process.exit(0);
    }
  });

  input.onSubmit = async () => {
    const value = input.plainText.trim();
    if (value === '/exit') {
      renderer.destroy();
      process.exit(0);
    }
    display.content = value ? `Echo: ${value}` : display.content;
    input.selectAll();
    input.deleteSelection();
    input.focus();
  };
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
