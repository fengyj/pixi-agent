import { afterAll, beforeAll, describe, it } from 'vitest';
import { Observation } from '../../../src';
const { getLogger, setupObservability, shutdownObservability } = Observation;

describe('observation integration smoke — write logs to ES', () => {
  beforeAll(async () => {
    await shutdownObservability();

    await setupObservability({
      transport: 'grpc',
      transportEndpoint: 'localhost:9937',
      logging: {
        serviceName: 'pixiagent-integration-test',
        serviceVersion: '0.1.0',
        outputToOtel: true,
        outputToConsole: true,
      },
    });
  });

  afterAll(async () => {
    await shutdownObservability();
  });

  it('writes multiple logs for manual ES inspection', async () => {
    const runId = `manual-es-smoke-${Date.now()}`;
    const log = getLogger('integration.manual.es.smoke', { runId, source: 'integration-test' });

    log.info({ step: 1, smoke: true }, 'manual es smoke info 1');
    log.info({ step: 2, smoke: true, kind: 'progress' }, 'manual es smoke info 2');
    log.warn({ step: 3, smoke: true, warning: 'expected-test-warning' }, 'manual es smoke warning');
    log.error({ step: 4, smoke: true, errorCode: 'SMOKE_TEST' }, 'manual es smoke error');

    // Give async log pipeline a moment before suite teardown flush.
    await new Promise<void>((resolve) => setTimeout(resolve, 300));
  });
});
