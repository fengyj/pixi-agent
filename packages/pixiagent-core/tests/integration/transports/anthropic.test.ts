/**
 * Integration tests for AnthropicTransport.
 *
 * Provider : ofox.ai  (https://api.ofox.ai/anthropic)
 * Model    : anthropic/claude-haiku-4.5
 * API key  : OFOXAI_API_KEY
 */

import { describe, it } from 'vitest';
import { AnthropicTransport } from '../../../src/transports/anthropic';
import { runStandardConversation } from './helpers';

const API_KEY = process.env.OFOXAI_API_KEY;
const BASE_URL = 'https://api.ofox.ai/anthropic';
const MODEL = 'anthropic/claude-haiku-4.5';

describe.skipIf(!API_KEY)('AnthropicTransport – ofox.ai', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY);

  it(
    'runs the standard 4-turn conversation including reasoning',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: true });
    },
    120_000,
  );
});
