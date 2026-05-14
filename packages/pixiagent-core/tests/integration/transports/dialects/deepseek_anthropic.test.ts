/**
 * Integration tests for AnthropicTransport + DeepSeekAnthropicDialectResolver.
 *
 * Provider : DeepSeek official  (https://api.deepseek.com/anthropic)
 * Model    : deepseek-v4-flash
 * API key  : DEEPSEEK_API_KEY
 *
 * Note: The DeepSeekAnthropicDialectResolver only enables extended thinking for
 * `deepseek-reasoner`. Since we're using `deepseek-v4-flash` here, Turn 4
 * (reasoning) is skipped.
 */

import { describe, it } from 'vitest';
import { AnthropicTransport } from '@pixiagent/core/transports/anthropic';
import { DeepSeekAnthropicDialectResolver } from '@pixiagent/core/transports/dialects/deepseek';
import { runStandardConversation } from '../helpers';

const API_KEY = process.env.DEEPSEEK_API_KEY;
const BASE_URL = 'https://api.deepseek.com/anthropic';
const MODEL = 'deepseek-v4-flash';

describe.skipIf(!API_KEY)('AnthropicTransport + DeepSeekAnthropicDialectResolver', () => {
  const transport = new AnthropicTransport(BASE_URL, API_KEY, new DeepSeekAnthropicDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning for non-reasoner model)',
    async () => {
      // deepseek-v4-flash does not expose thinking blocks via the Anthropic dialect;
      // reasoning is only available on deepseek-reasoner.
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    120_000,
  );
});
