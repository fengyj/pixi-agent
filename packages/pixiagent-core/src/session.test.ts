import { describe, expect, it } from 'vitest';
import { PendingMessageSchema } from './session';

describe('PendingMessageSchema', () => {
  it('generates pendingMessageId when missing', () => {
    const pending = PendingMessageSchema.parse({
      type: 'pending_message',
      role: 'user',
      content: 'hello',
    });

    expect(pending.pendingMessageId).toBeTypeOf('string');
    expect(pending.pendingMessageId).toHaveLength(12);
    expect(pending.pendingMessageId).not.toBe('');
  });

  it('preserves explicit pendingMessageId when provided', () => {
    const pending = PendingMessageSchema.parse({
      type: 'pending_message',
      role: 'user',
      content: 'hello',
      pendingMessageId: 'custom-id-123',
    });

    expect(pending.pendingMessageId).toBe('custom-id-123');
  });
});
