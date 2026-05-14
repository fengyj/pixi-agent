import { describe, expect, it } from 'vitest';
import { OpenRouterChatDialectResolver } from './openrouter';
import { DeepSeekChatDialectResolver } from './deepseek';
import { SessionMessage } from '../../message';

describe('chat dialect message manipulation', () => {
  it('OpenRouter keeps assistant text when adding thinking part', () => {
    const resolver = new OpenRouterChatDialectResolver();

    const msg: SessionMessage = {
      type: 'session_message',
      role: 'assistant',
      content: '2 + 2 = 4',
    };

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const rawMsg = {
      role: 'assistant',
      content: '2 + 2 = 4',
      reasoning: null,
      reasoning_details: [{ type: 'reasoning.text', text: ' to respond concisely as instructed.\n' }],
    } as any;
    /* eslint-enable @typescript-eslint/no-explicit-any */

    const out = resolver.manipulateMessage(msg, rawMsg);

    expect(Array.isArray(out.content)).toBe(true);
    expect(out.content).toEqual([
      { type: 'thinking', content: ' to respond concisely as instructed.\n', signature: undefined },
      { type: 'text', text: '2 + 2 = 4' },
    ]);
  });

  it('DeepSeek keeps assistant text when adding thinking part', () => {
    const resolver = new DeepSeekChatDialectResolver();

    const msg: SessionMessage = {
      type: 'session_message',
      role: 'assistant',
      content: '2 + 2 = 4',
    };

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const rawMsg = {
      role: 'assistant',
      content: '2 + 2 = 4',
      reasoning_content: 'First think quickly. ',
    } as any;
    /* eslint-enable @typescript-eslint/no-explicit-any */

    const out = resolver.manipulateMessage(msg, rawMsg);

    expect(Array.isArray(out.content)).toBe(true);
    expect(out.content).toEqual([
      { type: 'thinking', content: 'First think quickly. ' },
      { type: 'text', text: '2 + 2 = 4' },
    ]);
  });
});
