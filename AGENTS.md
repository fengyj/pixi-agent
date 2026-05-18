# Pixi Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

## Architecture

- `packages/pixi-agent-core`: core framework implementation.
- plugins and applications will be added as separate workspace packages.

## Goals

- Provide a clean core agent lifecycle and message-handling foundation.
- Support plugin hooks, extensible actions, and application integration.
- Keep package setup compatible with Bun monorepo workflows.

## Development Environment

- Use `bun` instead of `npm`.
- Use `vitest` instead of `jest` for testing. For example: `bun exec vitest run tests/integration/observation/observation.test.ts --config ../../vitest.integration.config.ts`.


# Custom Commands

## @grill-me

Interview me relentlessly about every aspect of this plan/design/requirement until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer or suggestions for me to choose.

Ask the questions one at a time.

If a question can be answered by exploring the codebase, explore the codebase instead.