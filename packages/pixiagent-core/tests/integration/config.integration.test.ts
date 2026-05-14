import { describe, expect, it } from 'vitest';
import { getIntegrationConfig } from './config';

describe('integration config', () => {
  it('reads and validates integration test config', () => {
    const config = getIntegrationConfig();

    expect(config.model).toBeTruthy();
    expect(typeof config.runLiveTests).toBe('boolean');
  });
});