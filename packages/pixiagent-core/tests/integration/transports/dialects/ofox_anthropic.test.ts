/**
 * Integration tests for AnthropicTransport + OfoxAnthropicDialectResolver.
 *
 * Provider : OFOX (https://api.ofox.ai/anthropic/v1/messages)
 * Model    : anthropic/claude-haiku-4.5
 * API key  : OFOXAI_API_KEY
 */

import { describe, it } from 'vitest';
import { AnthropicTransport } from '../../../../src/transports/anthropic';
import { OfoxAnthropicDialectResolver } from '../../../../src/transports/dialects/ofox';
import { assertBidirectionalConversion, runStandardConversation } from '../helpers';

const API_KEY = process.env.OFOXAI_API_KEY;
const BASE_URL = 'https://api.ofox.ai/anthropic/v1/messages';
const MODEL = 'anthropic/claude-haiku-4.5';

describe('AnthropicTransport + OfoxAnthropicDialectResolver conversion', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY, new OfoxAnthropicDialectResolver());

  it('converts between SessionMessage and raw message bidirectionally', () => {
    assertBidirectionalConversion(transport);
  });
});

describe.skipIf(!API_KEY)('AnthropicTransport + OfoxAnthropicDialectResolver', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY, new OfoxAnthropicDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    120_000,
  );
});
