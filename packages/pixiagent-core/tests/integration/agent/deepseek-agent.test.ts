/**
 * Integration test for PixiAgent with DeepSeek official Chat Completions endpoint.
 *
 * Provider : DeepSeek official  (https://api.deepseek.com)
 * Model    : deepseek-v4-flash
 * API key  : DEEPSEEK_API_KEY
 */

import { describe, expect, it } from 'vitest';
import { PixiAgent } from '@pixiagent/core/agent';
import { ApiModes } from '@pixiagent/core/message';
import { Session } from '@pixiagent/core/session';
import { ToolRegistry } from '@pixiagent/core/tool';

const API_KEY = process.env.DEEPSEEK_API_KEY;
const BASE_URL = 'https://api.deepseek.com';
const MODEL = 'deepseek-v4-flash';

describe.skipIf(!API_KEY)('PixiAgent + DeepSeek (deepseek-v4-flash)', () => {
  it(
    'executes one user turn and writes assistant response into the thread',
    async () => {
      const session = Session.create({
        modelOptions: {
          model: MODEL,
          baseUrl: BASE_URL,
          apiKey: API_KEY,
          apiMode: ApiModes.COMPLETIONS,
        },
      });

      const thread = Session.getDefaultThread(session);
      const agent = new PixiAgent(thread, new ToolRegistry(), {
        modelRequestTimeout: 120_000,
        maxIterations: 8,
      });

      const result = await agent.execute(
        {
          model: MODEL,
          baseUrl: BASE_URL,
          apiKey: API_KEY,
          apiMode: ApiModes.COMPLETIONS,
          toolChoice: 'none',
        },
        {
          type: 'session_message',
          role: 'user',
          content: 'Please answer in one short sentence: what is 2 + 2?',
        },
      );

      expect(result.stopReason).toBe('end_turn');
      expect(result.userMessageId).toBeDefined();
      expect(result.metadata?.PIXI_AGENT_THREAD_ID).toBe(thread.threadInfo.threadId);
      expect(thread.threadMessages.length).toBeGreaterThanOrEqual(2);

      const requestRaw = thread.threadMessages[thread.threadMessages.length - 2].rawMessage as {
        role?: string;
        content?: unknown;
      };
      const responseRaw = thread.threadMessages[thread.threadMessages.length - 1].rawMessage as {
        role?: string;
        content?: unknown;
        refusal?: string;
      };

      expect(requestRaw.role).toBe('user');
      expect(responseRaw.role).toBe('assistant');
      expect(responseRaw.refusal == null).toBe(true);

      const responseContent = responseRaw.content;
      if (typeof responseContent === 'string') {
        expect(responseContent.trim().length).toBeGreaterThan(0);
      } else if (Array.isArray(responseContent)) {
        const text = responseContent
          .filter(
            (part): part is { type: string; text: string } =>
              Boolean(part) &&
              typeof part === 'object' &&
              'type' in part &&
              (part as { type?: string }).type === 'text' &&
              'text' in part &&
              typeof (part as { text?: unknown }).text === 'string',
          )
          .map((part) => part.text)
          .join('')
          .trim();
        expect(text.length).toBeGreaterThan(0);
      } else {
        throw new Error('Unexpected assistant response content type');
      }

      expect(thread.threadInfo.modelOptions.model).toBe(MODEL);
      expect(thread.threadInfo.modelOptions.baseUrl).toBe(BASE_URL);
    },
    120_000,
  );
});
