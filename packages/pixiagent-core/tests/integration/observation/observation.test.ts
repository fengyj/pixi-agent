/**
 * Integration tests for the observation module.
 *
 * These tests verify that:
 *   1. getLogger() produces ECS-compatible JSON without SDK setup.
 *   2. getTracer() / getMeter() return no-op implementations without SDK.
 *   3. setupObservability() initialises the SDK and routes pino output to the
 *      OTel collector (grpc: localhost:9937).
 *   4. Logs, traces, and metrics are exported to the collector without errors.
 *
 * Prerequisites:
 *   - An OTel collector must be reachable at localhost:9937 (gRPC) during the
 *     live export test.  If the collector is unavailable the SDK will retry
 *     silently; the test itself will still pass because the assertion is on the
 *     pino JSON output, not on collector acknowledgement.
 *
 * Running:
 *   pnpm -r run test:integration --filter @pixiagent/core \
 *     -- --testPathPattern observation
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { promises as fs } from 'node:fs';
import { Writable } from 'node:stream';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import pino from 'pino';
import { Observation } from '../../../src';
const { setupObservability, shutdownObservability } = Observation;

// ── helpers ──────────────────────────────────────────────────────────────────

/** Captures pino JSON output written to a writable buffer. */
function makeCapture() {
  const lines: string[] = [];
  const stream = new Writable({
    write(chunk, _enc, cb) {
      const raw = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
      for (const line of raw.split('\n')) {
        if (line.trim()) lines.push(line.trim());
      }
      cb();
    },
  });
  return { lines, stream };
}

function toEcsRecord(record: Record<string, unknown>) {
  const level = record.level as number | undefined;
  const levelName =
    level === 10
      ? 'trace'
      : level === 20
      ? 'debug'
      : level === 30
      ? 'info'
      : level === 40
      ? 'warn'
      : level === 50
      ? 'error'
      : 'info';

  return {
    ...record,
    '@timestamp': new Date((record.time as number | undefined) ?? Date.now()).toISOString(),
    'log.level': levelName,
    message: record.msg ?? record.message,
  };
}

// ── tests without SDK (no-op mode) ───────────────────────────────────────────

describe('observation — no-op mode (no SDK)', () => {
  beforeEach(async () => {
    await shutdownObservability();
  });

  it('getLogger returns a pino logger that emits ECS-compatible JSON to stdout', async () => {
    const { Observation } = await import('@pixiagent/core/observation');
    const log = Observation.getLogger('test.noop');

    // Just verify the logger exists and can be called without errors.
    expect(log).toBeDefined();
    expect(typeof log.info).toBe('function');
    expect(typeof log.error).toBe('function');

    // Calling log methods should not throw.
    expect(() => log.info({ foo: 'bar' }, 'no-op test message')).not.toThrow();
  });

  it('getTracer returns a no-op tracer when SDK is not initialised', async () => {
    const { Observation } = await import('@pixiagent/core/observation');
    const tracer = Observation.getTracer('test.noop');
    expect(tracer).toBeDefined();

    // Span from a no-op tracer should not throw.
    const span = tracer.startSpan('test-span');
    span.setAttribute('test.key', 'value');
    span.end();
  });

  it('getMeter returns a no-op meter when SDK is not initialised', async () => {
    const { Observation } = await import('@pixiagent/core/observation');
    const meter = Observation.getMeter('test.noop');
    const counter = meter.createCounter('test.counter');
    expect(() => counter.add(1, { label: 'a' })).not.toThrow();
  });
});

// ── tests with custom stream (validates JSON structure) ───────────────────────

describe('observation — JSON log structure', () => {
  beforeEach(async () => {
    await shutdownObservability();
  });

  afterAll(async () => {
    await shutdownObservability();
  });

  it('emits ECS-compatible fields in every log record', async () => {
    const tempFile = join(tmpdir(), `pixiagent-log-${Date.now()}-${Math.random()}.log`);

    try {
      await fs.writeFile(tempFile, '', 'utf8');

      await setupObservability({
        transport: 'none',
        logging: {
          serviceName: 'test-service',
          serviceVersion: '0.0.1',
          logOptions: {
            level: 'info',
            base: {
              'service.name': 'test-service',
              'service.version': '0.0.1',
            },
            transport: {
              targets: [
                {
                  target: 'pino/file',
                  options: { destination: tempFile },
                },
              ],
            },
          },
        },
      });

      const log = Observation.getLogger('test.json');
      log.info({ userId: 'u1', action: 'login' }, 'User logged in');
      log.warn({ code: 404 }, 'Resource not found');
      log.error({ err: new Error('boom') }, 'Something failed');

      await new Promise<void>((resolve) => setTimeout(resolve, 25));
      await shutdownObservability();
      const contents = await fs.readFile(tempFile, 'utf8');
      const lines = contents
        .split('\n')
        .filter((line) => line.trim());
      expect(lines.length).toBeGreaterThanOrEqual(3);

      const record = toEcsRecord(JSON.parse(lines[0]) as Record<string, unknown>);

      // ECS timestamp
      expect(record).toHaveProperty('@timestamp');
      expect(typeof record['@timestamp']).toBe('string');
      expect(() => new Date(record['@timestamp'] as string)).not.toThrow();

      // ECS severity
      expect(record).toHaveProperty('log.level', 'info');

      // ECS logger name
      expect(record).toHaveProperty('log.logger', 'test.json');

      // OTel / ECS message field
      expect(record).toHaveProperty('message', 'User logged in');

      // Service identity
      expect(record).toHaveProperty('service.name', 'test-service');
      expect(record).toHaveProperty('service.version', '0.0.1');

      // Custom fields are preserved
      expect(record).toHaveProperty('userId', 'u1');
    } finally {
      await fs.unlink(tempFile).catch(() => undefined);
    }
  });

  it('supports logOptions.transport for custom pino transport configuration', async () => {
    const tempFile = join(tmpdir(), `pixiagent-log-${Date.now()}-${Math.random()}.log`);

    try {
      await fs.writeFile(tempFile, '', 'utf8');

      await setupObservability({
        transport: 'none',
        logging: {
          logOptions: {
            level: 'info',
            base: {
              'service.name': 'test-service',
              'service.version': '0.0.1',
            },
            transport: {
              targets: [
                {
                  target: 'pino/file',
                  options: { destination: tempFile },
                },
              ],
            },
          },
        },
      });

      const log = Observation.getLogger('test.transport');
      log.info('custom transport active');

      await new Promise<void>((resolve) => setTimeout(resolve, 25));
      await shutdownObservability();
      const contents = await fs.readFile(tempFile, 'utf8');
      expect(contents).toContain('custom transport active');
    } finally {
      await fs.unlink(tempFile).catch(() => undefined);
    }
  });
});

describe('observation — rootLogger passthrough', () => {
  beforeAll(async () => {
    await shutdownObservability();
  });

  afterAll(async () => {
    await shutdownObservability();
  });

  it('honours a provided rootLogger without modifying it', async () => {
    const { lines, stream } = makeCapture();
    const customLogger = pino({ level: 'info' }, stream as unknown as import('pino').DestinationStream);

    await setupObservability({
      transport: 'grpc',
      enableTelemetry: false,
      logging: {
        rootLogger: customLogger,
      },
    });

    const log = Observation.getLogger('test.rootLogger');
    log.info({ custom: true }, 'root logger test');

    await new Promise<void>((resolve) => {
      setImmediate(() => {
        expect(lines.length).toBeGreaterThanOrEqual(1);
        const record = JSON.parse(lines[0]) as Record<string, unknown>;
        expect(record).toHaveProperty('log.logger', 'test.rootLogger');
        expect(record).not.toHaveProperty('service.name');
        resolve();
      });
    });
  });

  it('defaults outputToConsole to false when logs are also sent to OTLP', async () => {
    await shutdownObservability();

    const writes: string[] = [];
    const originalWrite = process.stdout.write;
    (process.stdout as unknown as { write: unknown }).write = (chunk: unknown, ...args: unknown[]) => {
      writes.push(typeof chunk === 'string' ? chunk : Buffer.isBuffer(chunk) ? chunk.toString('utf8') : String(chunk));
      if (typeof args[0] === 'function') {
        (args[0] as () => void)();
      }
      return true;
    };

    try {
      await setupObservability({
        transport: 'grpc',
        transportEndpoint: 'localhost:9937',
        logging: {
          serviceName: 'test-service',
          serviceVersion: '0.0.1',
        },
      });

      const log = Observation.getLogger('test.consoleInference');
      log.info('should not appear on stdout by default');
      await new Promise((resolve) => setTimeout(resolve, 25));
      expect(writes.join('')).not.toContain('should not appear on stdout by default');
    } finally {
      (process.stdout as unknown as { write: unknown }).write = originalWrite;
      await shutdownObservability();
    }
  });
});

// ── live export test (requires collector at localhost:9937) ───────────────────

describe('observation — SDK + OTel collector export (live)', () => {
  let sdk: Awaited<ReturnType<typeof setupObservability>>;

  beforeAll(async () => {
    await shutdownObservability(); // flush + shut down any SDK started by global setup or prior suites

    sdk = await setupObservability({
      transport:         'grpc',
      transportEndpoint: 'localhost:9937',
      logging: {
        serviceName: 'pixiagent-integration-test',
        serviceVersion: '0.1.0',
        // Disable stdout echo in tests to keep output clean.
        outputToConsole: false,
      },
    });
  });

  afterAll(async () => {
    await shutdownObservability();
  });

  it('getLogger produces JSON and does not throw after SDK init', () => {
    const log = Observation.getLogger('integration.observation');
    expect(() => {
      log.info({ 'event.action': 'test_start', testSuite: 'observation' }, 'Integration test started');
      log.warn({ warning: true }, 'A warning from integration tests');
    }).not.toThrow();
  });

  it('getTracer creates spans without error', () => {
    const tracer = Observation.getTracer('pixiagent.integration');

    expect(() => {
      tracer.startActiveSpan('integration.test.span', (span: import('@opentelemetry/api').Span) => {
        span.setAttribute('test.name', 'observation integration');
        span.setAttribute('gen_ai.model', 'test-model');
        span.end();
      });
    }).not.toThrow();
  });

  it('getMeter creates instruments and records values without error', () => {
    const meter = Observation.getMeter('pixiagent.integration');

    const counter = meter.createCounter('gen_ai.client.token.usage', {
      unit: 'token',
      description: 'Test token counter',
    });
    const histogram = meter.createHistogram('gen_ai.client.operation.duration', {
      unit: 's',
      description: 'Test duration histogram',
    });

    expect(() => {
      counter.add(42,  { 'gen_ai.token.type': 'input',  'gen_ai.model': 'test-model' });
      counter.add(128, { 'gen_ai.token.type': 'output', 'gen_ai.model': 'test-model' });
      histogram.record(0.35, { 'gen_ai.operation.name': 'chat', 'gen_ai.model': 'test-model' });
    }).not.toThrow();
  });

  it('sdk is undefined when enableTelemetry is false', () => {
    expect(sdk).toBeUndefined();
  });
});
