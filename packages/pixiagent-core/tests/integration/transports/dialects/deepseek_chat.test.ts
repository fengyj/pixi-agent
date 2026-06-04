/**
 * Integration tests for ChatCompletionTransport + DeepSeekChatDialectResolver.
 *
 * Provider : DeepSeek official  (https://api.deepseek.com)
 * Model    : deepseek-v4-flash
 * API key  : DEEPSEEK_API_KEY
 *
 * DeepSeek's Chat Completions API returns reasoning via the `reasoning_content` field.
 * The dialect resolver maps it to ThinkingPart, so Turn 4 (reasoning) is enabled.
 */

import { describe, it } from 'vitest';
import { ChatCompletionTransport } from '../../../../src/transports/chat_completion';
import { DeepSeekChatDialectResolver } from '../../../../src/transports/dialects/deepseek';
import { runStandardConversation } from '../helpers';

const API_KEY = process.env.DEEPSEEK_API_KEY;
const BASE_URL = 'https://api.deepseek.com';
const MODEL = 'deepseek-v4-flash';
const REASONING_MODEL = 'deepseek-reasoner';

describe.skipIf(!API_KEY)('ChatCompletionTransport + DeepSeekChatDialectResolver', () => {
  const transport = new ChatCompletionTransport(BASE_URL, API_KEY, new DeepSeekChatDialectResolver());

  it(
    'runs the standard 4-turn conversation including reasoning',
    async () => {
      await runStandardConversation(transport, MODEL, {
        supportsReasoning: true,
        reasoningModel: REASONING_MODEL,
        reasoningInFinalMessage: false,
      });
    },
    120_000,
  );
});
