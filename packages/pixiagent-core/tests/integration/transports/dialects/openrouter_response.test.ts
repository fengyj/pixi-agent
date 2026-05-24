/**
 * Integration tests for ResponseTransport + OpenRouterResponseDialectResolver.
 *
 * Provider : OpenRouter (https://openrouter.ai/api/v1/responses)
 * Model    : openai/gpt-5.4-mini
 * API key  : OPENROUTER_API_KEY
 */

import { describe, it } from 'vitest';
import { ResponseTransport } from '../../../../src/transports/response';
import { OpenRouterResponseDialectResolver } from '../../../../src/transports/dialects/openrouter';
import { assertBidirectionalConversion, runStandardConversation } from '../helpers';

const API_KEY = process.env.OPENROUTER_API_KEY;
const BASE_URL = 'https://openrouter.ai/api/v1/responses';
const MODEL = 'openai/gpt-5.4-mini';

describe('ResponseTransport + OpenRouterResponseDialectResolver conversion', () => {
  const transport = new ResponseTransport(BASE_URL, API_KEY, new OpenRouterResponseDialectResolver());

  it('converts between SessionMessage and raw message bidirectionally', () => {
    assertBidirectionalConversion(transport);
  });
});

describe.skipIf(!API_KEY)('ResponseTransport + OpenRouterResponseDialectResolver', () => {
  const transport = new ResponseTransport(BASE_URL, API_KEY, new OpenRouterResponseDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    180_000,
  );
});
