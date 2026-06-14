/**
 * Integration test for AgentThreadRunner with DeepSeek official Chat Completions endpoint.
 *
 * Provider : DeepSeek official  (https://api.deepseek.com)
 * Model    : deepseek-v4-flash
 * API key  : DEEPSEEK_API_KEY
 */

import { describe, expect, it } from 'vitest';
import { AgentThreadRunner, ApiModes, Session } from '../../../src';
import { AgentEventEmitter } from '../../../src/event';
import { ToolRegistry } from '../../../src/tools';

const API_KEY = process.env.DEEPSEEK_API_KEY;
const BASE_URL = 'https://api.deepseek.com';
const MODEL = 'deepseek-v4-flash';

describe.skipIf(!API_KEY)('AgentThreadRunner + DeepSeek (deepseek-v4-flash)', () => {
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
      const agent = new AgentThreadRunner(thread, new ToolRegistry(), {
        eventEmitter: new AgentEventEmitter(),
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

      if (result.action !== 'execution') {
        throw new Error(`Expected execution result, got ${result.action}`);
      }

      expect(result.action).toBe('execution');
      expect(result.stopReason).toBe('end_turn');
      expect(result.messages).toBeDefined();
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
      const extractedText = (() => {
        if (typeof responseContent === 'string') {
          return responseContent.trim();
        }

        if (Array.isArray(responseContent)) {
          return responseContent
            .map((part) => {
              if (typeof part === 'string') return part;
              if (part && typeof part === 'object') {
                if ('text' in part && typeof (part as { text?: unknown }).text === 'string') {
                  return (part as { text: string }).text;
                }
                if ('content' in part && typeof (part as { content?: unknown }).content === 'string') {
                  return (part as { content: string }).content;
                }
              }
              return '';
            })
            .join('')
            .trim();
        }

        if (responseContent && typeof responseContent === 'object') {
          if ('text' in responseContent && typeof (responseContent as { text?: unknown }).text === 'string') {
            return (responseContent as { text: string }).text.trim();
          }
          if ('content' in responseContent && typeof (responseContent as { content?: unknown }).content === 'string') {
            return (responseContent as { content: string }).content.trim();
          }
        }

        return '';
      })();

      expect(extractedText.length).toBeGreaterThan(0);

      expect(thread.threadInfo.modelOptions.model).toBe(MODEL);
      expect(thread.threadInfo.modelOptions.baseUrl).toBe(BASE_URL);
    },
    120_000,
  );
});
