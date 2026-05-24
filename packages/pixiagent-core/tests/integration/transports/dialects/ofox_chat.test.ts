/**
 * Integration tests for ChatCompletionTransport + OfoxChatDialectResolver.
 *
 * Provider : OFOX (https://api.ofox.ai/v1/chat/completions)
 * Model    : openai/gpt-5.4-nano
 * API key  : OFOXAI_API_KEY
 */

import { describe, it } from 'vitest';
import { ChatCompletionTransport } from '../../../../src/transports/chat_completion';
import { OfoxChatDialectResolver } from '../../../../src/transports/dialects/ofox';
import { assertBidirectionalConversion, runStandardConversation } from '../helpers';

const API_KEY = process.env.OFOXAI_API_KEY;
const BASE_URL = 'https://api.ofox.io/v1/chat/completions';
const MODEL = 'openai/gpt-5.4-nano';

describe('ChatCompletionTransport + OfoxChatDialectResolver conversion', () => {
  const transport = new ChatCompletionTransport(BASE_URL, API_KEY, new OfoxChatDialectResolver());

  it('converts between SessionMessage and raw message bidirectionally', () => {
    assertBidirectionalConversion(transport);
  });
});

describe.skipIf(!API_KEY)('ChatCompletionTransport + OfoxChatDialectResolver', () => {
  const transport = new ChatCompletionTransport(BASE_URL, API_KEY, new OfoxChatDialectResolver());

  it(
    'runs the standard 3-turn conversation (no reasoning)',
    async () => {
      await runStandardConversation(transport, MODEL, { supportsReasoning: false });
    },
    120_000,
  );
});
