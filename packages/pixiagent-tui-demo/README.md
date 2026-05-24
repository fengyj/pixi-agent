# @pixiagent/tui-demo

A simple interactive CLI chat demo based on PixiAgent.

## Features

- Interactive loop like a coding assistant terminal.
- Model configuration loaded from `.env`.
- Displays the final assistant message for each turn.
- Multi-line input (`Enter` newline, `Ctrl+Enter` send).
- Line-based message scrolling with keyboard shortcuts.
- Mouse support is temporarily disabled for terminal compatibility.
- Optional observability via `Observation.setupObservability`.
- Includes a backend interface so UI/backend transport can later be swapped to ACP.

## Environment Setup

Create `.env` in this package directory by copying `.env.example`.

Required variables:

- `PIXIA_MODEL`
- `PIXIA_BASE_URL`
- `PIXIA_API_KEY_ENV` (name of the env var that stores the real API key)
- The real API key variable itself (for example `DEEPSEEK_API_KEY`)

Optional model variables:

- `PIXIA_SYSTEM_PROMPT` (passed to `agent.execute` via `modelOptions.systemPrompt`)

Optional observability variables:

- `PIXIA_OTEL_ENABLED=true|false`
- `PIXIA_OTEL_TRANSPORT=grpc|http|none`
- `PIXIA_OTEL_ENDPOINT` (required when transport is `grpc` or `http`)
- `PIXIA_OTEL_ENABLE_TELEMETRY=true|false`
- `PIXIA_OTEL_SERVICE_NAME`
- `PIXIA_OTEL_SERVICE_VERSION`
- `PIXIA_OTEL_OUTPUT_TO_CONSOLE=true|false`
- `PIXIA_OTEL_OUTPUT_TO_OTEL=true|false`

## Run

From repository root:

- `bun run --filter @pixiagent/tui-demo start`

Or from this package directory:

- `bun run start`

Commands:

- `/exit`: quit

Navigation:

- `PageUp` / `Ctrl+Up`: scroll up by half page
- `PageDown` / `Ctrl+Down`: scroll down by half page
- `Ctrl+U`: scroll up by one line
- `Ctrl+D`: scroll down by one line
- `Ctrl+Home`: jump to oldest visible point
- `Ctrl+End`: jump to latest messages

Input:

- `Enter`: insert newline
- `Ctrl+Enter`: submit current message
- `Ctrl+C`: graceful shutdown and exit

## Observability

When `PIXIA_OTEL_ENABLED=true`, the CLI calls `Observation.setupObservability(...)` at startup and `Observation.shutdownObservability()` on exit.

To avoid TUI rendering corruption, this demo disables observability console output while TUI is running. Keep `PIXIA_OTEL_OUTPUT_TO_CONSOLE=false` and use OTEL export instead.

Example:

```env
PIXIA_OTEL_ENABLED=true
PIXIA_OTEL_TRANSPORT=http
PIXIA_OTEL_ENDPOINT=http://localhost:4318
PIXIA_OTEL_ENABLE_TELEMETRY=true
PIXIA_OTEL_OUTPUT_TO_CONSOLE=false
PIXIA_OTEL_OUTPUT_TO_OTEL=true
```

## Troubleshooting

If you see escape-sequence gibberish (for example `^[[...`) inside the input area:

- Set `PIXIA_OTEL_OUTPUT_TO_CONSOLE=false`.
- Prefer running from package directory: `cd packages/pixiagent-tui-demo && bun run start`.
- If needed, restart terminal session and run again.
- OpenTUI official compatibility switches used by this demo:
	- `OPENTUI_FORCE_EXPLICIT_WIDTH=false` (skip OSC 66 width queries)
	- `OTUI_USE_CONSOLE=false` (disable OpenTUI console capture)

## Note on System Prompt

This demo does not send a `system` role message in chat history. If you need system instructions, set `PIXIA_SYSTEM_PROMPT`; it will be passed through `agent.execute(..., modelOptions)` using `systemPrompt`.
