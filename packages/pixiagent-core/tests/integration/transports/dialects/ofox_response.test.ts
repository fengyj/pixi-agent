/**
 * Integration tests for ResponseTransport + OfoxResponseDialectResolver.
 *
 * Provider : OFOX (https://api.ofox.ai/v1/responses)
 * Model    : openai/gpt-5.4-nano
 * API key  : OFOXAI_API_KEY
 */

import { describe, it } from 'vitest';
import { ResponseTransport } from '../../../../src/transports/response';
import { OfoxResponseDialectResolver } from '../../../../src/transports/dialects/ofox';
import { assertBidirectionalConversion, runStandardConversation } from '../helpers';

const API_KEY = process.env.OFOXAI_API_KEY;
const BASE_URL = 'https://api.ofox.ai/v1/responses';
const MODEL = 'openai/gpt-5.4-nano';

describe('ResponseTransport + OfoxResponseDialectResolver conversion', () => {
  const transport = new ResponseTransport(BASE_URL, API_KEY, new OfoxResponseDialectResolver());

  it('converts between SessionMessage and raw message bidirectionally', () => {
    assertBidirectionalConversion(transport);
  });
});

describe.skipIf(!API_KEY)('ResponseTransport + OfoxResponseDialectResolver', () => {
  const transport = new ResponseTransport(BASE_URL, API_KEY, new OfoxResponseDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    120_000,
  );
});
