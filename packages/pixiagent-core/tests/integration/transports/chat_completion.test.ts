/**
 * Integration tests for ChatCompletionTransport (no dialect).
 *
 * Provider : ofox.ai  (https://api.ofox.ai/v1)
 * Model    : openai/gpt-5.4-nano
 * API key  : OFOXAI_API_KEY
 *
 * gpt-5.4-nano does not expose reasoning tokens, so Turn 4 (reasoning) is skipped.
 */

import { describe, it } from 'vitest';
import { ChatCompletionTransport } from '@pixiagent/core/transports/chat_completion';
import { runStandardConversation } from './helpers';

const API_KEY = process.env.OFOXAI_API_KEY;
const BASE_URL = 'https://api.ofox.ai/v1';
const MODEL = 'openai/gpt-5.4-nano';

describe.skipIf(!API_KEY)('ChatCompletionTransport – ofox.ai', () => {
  const transport = new ChatCompletionTransport(BASE_URL, API_KEY);

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    120_000,
  );
});
