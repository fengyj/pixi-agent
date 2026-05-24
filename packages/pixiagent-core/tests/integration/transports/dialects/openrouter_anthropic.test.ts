/**
 * Integration tests for AnthropicTransport + OpenRouterAnthropicDialectResolver.
 *
 * Provider : OpenRouter (https://openrouter.ai/api/v1/messages)
 * Model    : anthropic/claude-haiku-4.5
 * API key  : OPENROUTER_API_KEY
 */

import { describe, it } from 'vitest';
import { AnthropicTransport } from '../../../../src/transports/anthropic';
import { OpenRouterAnthropicDialectResolver } from '../../../../src/transports/dialects/openrouter';
import { assertBidirectionalConversion, runStandardConversation } from '../helpers';

const API_KEY = process.env.OPENROUTER_API_KEY;
const BASE_URL = 'https://openrouter.ai/api/v1/messages';
const MODEL = 'anthropic/claude-haiku-4.5';

describe('AnthropicTransport + OpenRouterAnthropicDialectResolver conversion', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY, new OpenRouterAnthropicDialectResolver());

  it('converts between SessionMessage and raw message bidirectionally', () => {
    assertBidirectionalConversion(transport);
  });
});

describe.skipIf(!API_KEY)('AnthropicTransport + OpenRouterAnthropicDialectResolver', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY, new OpenRouterAnthropicDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    180_000,
  );
});
