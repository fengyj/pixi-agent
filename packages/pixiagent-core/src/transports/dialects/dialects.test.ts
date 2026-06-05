import { describe, expect, it } from 'vitest';
import { OpenRouterChatDialectResolver } from './openrouter';
import { DeepSeekChatDialectResolver } from './deepseek';
import { ChatCompletionApiMessage, SessionMessage } from '../../message';
import { StreamDataExtractor } from '../base';

describe('chat dialect message manipulation', () => {
  it('OpenRouter keeps assistant text when adding thinking part', () => {
    const resolver = new OpenRouterChatDialectResolver();

    const msg: SessionMessage = {
      messageId: '',
      type: 'session_message',
      role: 'assistant',
      content: '2 + 2 = 4',
    };

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const rawMsg: ChatCompletionApiMessage = {
      messageId: '',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: '2 + 2 = 4',
        reasoning: null,
        reasoning_details: [{ type: 'reasoning.text', text: ' to respond concisely as instructed.\n' }],
      } as any,
    };
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
      messageId: '',
      type: 'session_message',
      role: 'assistant',
      content: '2 + 2 = 4',
    };

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const rawMsg: ChatCompletionApiMessage = {
      messageId: '',
      type: 'chat_completion_api_message',
      role: 'assistant',
      content: {
        role: 'assistant',
        content: '2 + 2 = 4',
        reasoning_content: 'First think quickly. ',
      } as any,
    };
    /* eslint-enable @typescript-eslint/no-explicit-any */

    const out = resolver.manipulateMessage(msg, rawMsg);

    expect(Array.isArray(out.content)).toBe(true);
    expect(out.content).toEqual([
      { type: 'thinking', content: 'First think quickly. ' },
      { type: 'text', text: '2 + 2 = 4' },
    ]);
  });

  it('OpenRouter extracts reasoning.text delta', async () => {
    const resolver = new OpenRouterChatDialectResolver();
    const streamDataExtractor = new StreamDataExtractor({
      choices: [{ message: {} as Record<string, unknown> }],
    });

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const delta = {
      reasoning_details: [{ index: 0, type: 'reasoning.text', text: 'step one. ' }],
    } as any;
    /* eslint-enable @typescript-eslint/no-explicit-any */

    await resolver.extractFromDelta('reasoning', delta, streamDataExtractor);
    expect(
      (streamDataExtractor.accumulatedData.choices[0].message as { reasoning_details?: Array<{ text?: string }> })
        .reasoning_details?.[0]?.text,
    ).toBe('step one. ');
  });

  it('OpenRouter extracts reasoning.summary delta', async () => {
    const resolver = new OpenRouterChatDialectResolver();
    const streamDataExtractor = new StreamDataExtractor({
      choices: [{ message: {} as Record<string, unknown> }],
    });

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const delta = {
      reasoning_details: [{ index: 0, type: 'reasoning.summary', summary: 'summary chunk. ' }],
    } as any;
    /* eslint-enable @typescript-eslint/no-explicit-any */

    await resolver.extractFromDelta('reasoning', delta, streamDataExtractor);
    expect(
      (streamDataExtractor.accumulatedData.choices[0].message as { reasoning_details?: Array<{ summary?: string }> })
        .reasoning_details?.[0]?.summary,
    ).toBe('summary chunk. ');
  });
});
