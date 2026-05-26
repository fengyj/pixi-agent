# pixiagent

Pixi Agent is a TypeScript monorepo for building an extensible agent system.

## Packages

- `packages/pixiagent-core`: core agent framework used by the Pixi agent system.
- `packages/pixiagent-tui-demo`: interactive CLI chat demo powered by PixiAgent.

## Setup

Install dependencies:

```bash
pnpm install
```

Build all packages:

```bash
pnpm -r run build
```

Run lint across the monorepo:

```bash
pnpm -r run lint
```

Run tests:

```bash
pnpm -r run test:unit
```

## Project structure

- `packages/pixiagent-core`: core framework implementation, including agent lifecycle, message handling, and plugin hooks.
- `packages/pixiagent-tui-demo`: command-line chat demo with environment-based model config.
- `agent.md`: design notes and project architecture.
