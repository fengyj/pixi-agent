# pixiagent

Pixi Agent is a TypeScript Bun monorepo for building an extensible agent system.

## Packages

- `packages/pixiagent-core`: core agent framework used by the Pixi agent system.
- `packages/pixiagent-tui-demo`: interactive CLI chat demo powered by PixiAgent.

## Setup

Install dependencies:

```bash
bun install
```

Build all packages:

```bash
bun workspaces run build
```

Run lint across the monorepo:

```bash
bun workspaces run lint
```

Run tests:

```bash
bun workspaces run test:unit
```

## Project structure

- `packages/pixiagent-core`: core framework implementation, including agent lifecycle, message handling, and plugin hooks.
- `packages/pixiagent-tui-demo`: command-line chat demo with environment-based model config.
- `agent.md`: design notes and project architecture.
