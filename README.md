# pixiagent

Pixi Agent is a TypeScript pnpm monorepo for building an extensible agent system.

## Packages

- `packages/pixiagent-core`: core agent framework used by the Pixi agent system.

## Setup

Install dependencies:

```bash
pnpm install
```

Build all packages:

```bash
pnpm run build
```

Run lint across the monorepo:

```bash
pnpm run lint
```

Run tests:

```bash
pnpm run test
```

## Project structure

- `packages/pixiagent-core`: core framework implementation, including agent lifecycle, message handling, and plugin hooks.
- `agent.md`: design notes and project architecture.
