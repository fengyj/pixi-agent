import { describe, expect, it } from 'vitest';
import { ApiModes } from '../message';
import { Transport } from './index';

describe('Transport.GlobalApiModeResolverRegistry', () => {
  it('resolves OpenRouter base URL to completions mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-4o-mini',
      'https://openrouter.ai/api/v1',
    );

    expect(resolved).toEqual(['https://openrouter.ai/api/v1', ApiModes.COMPLETIONS]);
  });

  it('infers DeepSeek base URL when apiMode is anthropic and baseUrl is absent', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'deepseek-reasoner',
      undefined,
      ApiModes.ANTHROPIC,
    );

    expect(resolved).toEqual(['https://api.deepseek.com/anthropic', ApiModes.ANTHROPIC]);
  });

  it('falls back to explicit apiMode when no resolver recognizes the baseUrl', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'unknown-model',
      'https://example.com/api/v1',
      ApiModes.RESPONSE,
    );

    expect(resolved).toEqual(['https://example.com/api/v1', ApiModes.RESPONSE]);
  });

  it('returns undefined when neither apiMode nor resolver match exists', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'unknown-model',
      'https://example.com/api/v1',
    );

    expect(resolved).toBeUndefined();
  });
});