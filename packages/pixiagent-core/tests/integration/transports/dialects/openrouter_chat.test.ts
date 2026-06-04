/**
 * Integration tests for ChatCompletionTransport + OpenRouterChatDialectResolver.
 *
 * Provider : OpenRouter  (https://openrouter.ai/api/v1)
 * API key  : OPENROUTER_API_KEY
 *
 * Models excluded per request: openai/*, anthropic/*, google/*, moonshotai/kimi-latest.
 *
 * Reasoning models (Turn 4 enabled):
 *   - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
 *   - qwen/qwen3.5-flash-02-23
 *
 * Non-reasoning models (Turn 4 skipped):
 *   - x-ai/grok-4.1-fast
 *   - tencent/hy3-preview:free
 *   - minimax/minimax-m2.5:free
 */

import { describe, it } from 'vitest';
import { ChatCompletionTransport } from '../../../../src/transports/chat_completion';
import { OpenRouterChatDialectResolver } from '../../../../src/transports/dialects/openrouter';
import { runStandardConversation } from '../helpers';

const API_KEY = process.env.OPENROUTER_API_KEY;
const BASE_URL = 'https://openrouter.ai/api/v1';

const MODELS: { model: string; supportsReasoning: boolean }[] = [
  { model: 'x-ai/grok-4.1-fast',                                   supportsReasoning: false },
  { model: 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',   supportsReasoning: true  },
  { model: 'qwen/qwen3.5-flash-02-23',                             supportsReasoning: true  },
  { model: 'tencent/hy3-preview:free',                             supportsReasoning: false },
  { model: 'minimax/minimax-m2.5:free',                            supportsReasoning: false },
];

describe.skipIf(!API_KEY)('ChatCompletionTransport + OpenRouterChatDialectResolver', () => {
  it.each(MODELS)(
    '$model (reasoning=$supportsReasoning)',
    async ({ model, supportsReasoning }) => {
      const transport = new ChatCompletionTransport(
        BASE_URL,
        API_KEY,
        new OpenRouterChatDialectResolver(),
      );
      await runStandardConversation(transport, model, { supportsReasoning });
    },
    180_000,
  );
});
