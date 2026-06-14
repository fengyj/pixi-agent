import { afterEach, describe, expect, it, vi } from 'vitest';

const executeMock = vi.fn();

vi.mock('./agent-loop', () => {
  class MockAgentExecutionLoop {
    public execute = executeMock;
  }

  return {
    AgentExecutionLoop: MockAgentExecutionLoop,
  };
});

import { AgentThreadRunner } from './agent-thread-runner';
import { ApiModes } from './message';
import * as Transport from './transports';

const resolveSpy = vi.spyOn(Transport.GlobalApiModeResolverRegistry, 'resolve');

afterEach(() => {
  vi.restoreAllMocks();
});

describe('AgentThreadRunner execute', () => {
  it('queues concurrent execute calls and runs them serially', async () => {
    resolveSpy.mockReturnValue(['https://example.com/api/v1', ApiModes.RESPONSE]);

    executeMock.mockReset();
    executeMock.mockImplementation(async () => ({
      action: 'execution',
      stopReason: 'end_turn',
      messages: [],
    } as never));

    const agent = new AgentThreadRunner(
      {
        session: { sessionId: 'session-1' },
        threadInfo: {
          threadId: 'thread-1',
          modelOptions: { model: 'gpt-test', apiMode: ApiModes.RESPONSE },
        },
      } as never,
      {} as never,
    );

    let releaseFirst = () => {};
    const firstStarted = new Promise<void>((resolve) => {
      executeMock.mockImplementationOnce(async () => {
        resolve();
        await new Promise<void>((release) => {
          releaseFirst = release;
        });
        return {
          action: 'execution',
          stopReason: 'end_turn',
          messages: [],
        } as never;
      });
    });

    const firstPromise = agent.execute(
      { model: 'gpt-test', apiMode: ApiModes.RESPONSE },
      { role: 'user', content: 'first' } as never,
    );

    await firstStarted;

    const secondPromise = agent.execute(
      { model: 'gpt-test', apiMode: ApiModes.RESPONSE },
      { role: 'user', content: 'second' } as never,
    );

    await Promise.resolve();
    expect(executeMock).toHaveBeenCalledTimes(1);
    await expect(secondPromise).resolves.toEqual({ action: 'steering' });

    releaseFirst();

    await expect(firstPromise).resolves.toEqual({ action: 'execution', stopReason: 'end_turn', messages: [] });
    await expect(secondPromise).resolves.toEqual({ action: 'steering' });
    expect(executeMock).toHaveBeenCalledTimes(2);
  });
});
