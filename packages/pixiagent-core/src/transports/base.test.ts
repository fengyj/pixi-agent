import { describe, expect, it } from 'vitest';
import { ApiModes } from '../message';
import * as Transport from './index';

describe('Transport.GlobalApiModeResolverRegistry', () => {
  it('resolves OpenRouter base URL to completions mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-4o-mini',
      'https://openrouter.ai/api/v1',
    );

    expect(resolved).toEqual(['https://openrouter.ai/api/v1', ApiModes.COMPLETIONS]);
  });

  it('resolves OpenRouter responses endpoint URL to response mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-5.4-nano',
      'https://openrouter.ai/api/v1/responses',
    );

    expect(resolved).toEqual(['https://openrouter.ai/api/v1/responses', ApiModes.RESPONSE]);
  });

  it('resolves OpenRouter anthropic messages endpoint URL to anthropic mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'anthropic/claude-sonnet-4',
      'https://openrouter.ai/api/v1/messages',
    );

    expect(resolved).toEqual(['https://openrouter.ai/api/v1/messages', ApiModes.ANTHROPIC]);
  });

  it('resolves OFOX chat completions endpoint URL to completions mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-5.4-mini',
      'https://api.ofox.ai/v1/chat/completions',
    );

    expect(resolved).toEqual(['https://api.ofox.ai/v1/chat/completions', ApiModes.COMPLETIONS]);
  });

  it('resolves OFOX .io chat completions endpoint URL to completions mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-5.4-mini',
      'https://api.ofox.io/v1/chat/completions',
    );

    expect(resolved).toEqual(['https://api.ofox.io/v1/chat/completions', ApiModes.COMPLETIONS]);
  });

  it('resolves OFOX responses endpoint URL to response mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-5.4-mini',
      'https://api.ofox.ai/v1/responses',
    );

    expect(resolved).toEqual(['https://api.ofox.ai/v1/responses', ApiModes.RESPONSE]);
  });

  it('resolves OFOX .io responses endpoint URL to response mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'openai/gpt-5.4-mini',
      'https://api.ofox.io/v1/responses',
    );

    expect(resolved).toEqual(['https://api.ofox.io/v1/responses', ApiModes.RESPONSE]);
  });

  it('resolves OFOX anthropic messages endpoint URL to anthropic mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'anthropic/claude-sonnet-4.6',
      'https://api.ofox.ai/anthropic/v1/messages',
    );

    expect(resolved).toEqual([
      'https://api.ofox.ai/anthropic/v1/messages',
      ApiModes.ANTHROPIC,
    ]);
  });

  it('resolves OFOX .io anthropic messages endpoint URL to anthropic mode', () => {
    const resolved = Transport.GlobalApiModeResolverRegistry.resolve(
      'anthropic/claude-sonnet-4.6',
      'https://api.ofox.io/anthropic/v1/messages',
    );

    expect(resolved).toEqual([
      'https://api.ofox.io/anthropic/v1/messages',
      ApiModes.ANTHROPIC,
    ]);
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