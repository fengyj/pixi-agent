/**
 * Live integration tests for PixiAgent timeout and abort handling.
 *
 * No mocks are used. These cases hit real provider SDKs/endpoints.
 */

import { describe, expect, it } from 'vitest';
import { PixiAgent, PixiAgentOptions } from '@pixiagent/core/agent';
import { ModelRequestTimeoutError } from '@pixiagent/core/errors';
import { ApiModes, SessionMessage } from '@pixiagent/core/message';
import { Session, SessionThread } from '@pixiagent/core/session';
import { ToolRegistry } from '@pixiagent/core/tool';

type LiveCase = {
  name: string;
  model: string;
  baseUrl: string;
  apiKey: string;
  apiMode: ApiModes;
};

const LIVE_CASES: LiveCase[] = [
  {
    name: 'ChatCompletion / OFOX / openai-gpt-5.4-nano',
    model: 'openai/gpt-5.4-nano',
    baseUrl: 'https://api.ofox.ai/v1',
    apiKey: process.env.OFOXAI_API_KEY ?? '',
    apiMode: ApiModes.COMPLETIONS,
  },
  {
    name: 'Anthropic / OFOX / claude-haiku-4.5',
    model: 'anthropic/claude-haiku-4.5',
    baseUrl: 'https://api.ofox.ai/anthropic',
    apiKey: process.env.OFOXAI_API_KEY ?? '',
    apiMode: ApiModes.ANTHROPIC,
  },
  {
    name: 'ChatCompletion / DeepSeek / deepseek-v4-flash',
    model: 'deepseek-v4-flash',
    baseUrl: 'https://api.deepseek.com',
    apiKey: process.env.DEEPSEEK_API_KEY ?? '',
    apiMode: ApiModes.COMPLETIONS,
  },
  {
    name: 'ChatCompletion / OpenRouter / x-ai-grok-4.1-fast',
    model: 'x-ai/grok-4.1-fast',
    baseUrl: 'https://openrouter.ai/api/v1',
    apiKey: process.env.OPENROUTER_API_KEY ?? '',
    apiMode: ApiModes.COMPLETIONS,
  },
];

const ENABLED_CASES = LIVE_CASES.filter((c) => c.apiKey.length > 0);

function createAgent(
  liveCase: LiveCase,
  options?: PixiAgentOptions,
): { agent: PixiAgent; thread: SessionThread } {
  const session = Session.create({
    modelOptions: {
      model: liveCase.model,
      baseUrl: liveCase.baseUrl,
      apiKey: liveCase.apiKey,
      apiMode: liveCase.apiMode,
    },
  });

  const thread = Session.getDefaultThread(session);
  const agent = new PixiAgent(thread, new ToolRegistry(), options);
  return { agent, thread };
}

function buildUserMessage(content: string): SessionMessage {
  return {
    type: 'session_message',
    role: 'user',
    content,
  };
}

describe.skipIf(ENABLED_CASES.length === 0)('PixiAgent live timeout + abort handling', () => {
  it.each(ENABLED_CASES)(
    '$name - surfaces timeout error from real SDK call',
    async (liveCase) => {
      const { agent } = createAgent(liveCase, {
        modelRequestTimeout: 100,
        maxIterations: 2,
        maxModelRequestRetries: 0,
      });

      const message = buildUserMessage(
        'Write a detailed answer with many paragraphs about distributed systems clocks and include examples.',
      );

      let caught: unknown;
      try {
        await agent.execute(
          {
            model: liveCase.model,
            baseUrl: liveCase.baseUrl,
            apiKey: liveCase.apiKey,
            apiMode: liveCase.apiMode,
            toolChoice: 'none',
          },
          message,
        );
      } catch (error) {
        caught = error;
      }

      expect(caught).toBeDefined();
      expect(caught instanceof Error).toBe(true);
      expect(
        caught instanceof ModelRequestTimeoutError ||
          (caught as Error).name.toLowerCase().includes('timeout') ||
          (caught as Error).name.toLowerCase().includes('abort') ||
          (caught as Error).message.toLowerCase().includes('aborted') ||
          (caught as Error).message.toLowerCase().includes('timed out') ||
          (caught as Error).message.toLowerCase().includes('timeout'),
      ).toBe(true);
    },
    120_000,
  );

  it.each(ENABLED_CASES)(
    '$name - returns cancelled when interrupted (abort path)',
    async (liveCase) => {
      const { agent, thread } = createAgent(liveCase, {
        modelRequestTimeout: 120_000,
        maxIterations: 2,
      });

      const executePromise = agent.execute(
        {
          model: liveCase.model,
          baseUrl: liveCase.baseUrl,
          apiKey: liveCase.apiKey,
          apiMode: liveCase.apiMode,
          toolChoice: 'none',
        },
        buildUserMessage(
          'Think carefully and produce a long response with many sections about event sourcing trade-offs.',
        ),
      );

      agent.interrupt('integration-test abort');

      const result = await executePromise;

      expect(result.stopReason).toBe('cancelled');
      expect(result.userMessageId).toBeDefined();
      expect(result.metadata?.PIXI_AGENT_THREAD_ID).toBe(thread.threadInfo.threadId);
    },
    120_000,
  );
});
