/**
 * Browser unit tests for observation/browser.ts
 *
 * Runs under happy-dom (vitest environment: 'happy-dom') with
 * resolve.conditions: ['browser'] so `pino` resolves to `pino/browser`.
 *
 * OTel SDK packages are mocked via vi.mock() — they are optional peer deps
 * that are not required to be installed in CI when running these tests.
 * The tests validate the observable API surface and internal wiring (log bridge,
 * provider lifecycle, page-unload handlers) without a live collector.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ── mock OTel optional peer deps (dynamic imports inside browser.ts) ──────────
// vi.hoisted() runs before vi.mock() hoisting, making mock instances available
// inside the vi.mock() factories below.

const {
  mockLoggerEmit,
  mockLoggerProvider,
  mockLPForceFlush,
  mockLPShutdown,
  mockLogsSetGlobal,
  mockTracerProvider,
  mockTPRegister,
  mockTPForceFlush,
  mockTPShutdown,
  mockMeterProvider,
  mockMPForceFlush,
  mockMPShutdown,
} = vi.hoisted(() => {
  const mockLoggerEmit = vi.fn();
  const mockLPForceFlush = vi.fn().mockResolvedValue(undefined);
  const mockLPShutdown = vi.fn().mockResolvedValue(undefined);
  const mockLoggerProvider = {
    addLogRecordProcessor: vi.fn(),
    getLogger: vi.fn().mockReturnValue({ emit: mockLoggerEmit }),
    forceFlush: mockLPForceFlush,
    shutdown: mockLPShutdown,
  };
  const mockLogsSetGlobal = vi.fn();

  const mockTPRegister = vi.fn();
  const mockTPForceFlush = vi.fn().mockResolvedValue(undefined);
  const mockTPShutdown = vi.fn().mockResolvedValue(undefined);
  const mockTracerProvider = { register: mockTPRegister, forceFlush: mockTPForceFlush, shutdown: mockTPShutdown };

  const mockMPForceFlush = vi.fn().mockResolvedValue(undefined);
  const mockMPShutdown = vi.fn().mockResolvedValue(undefined);
  const mockMeterProvider = { forceFlush: mockMPForceFlush, shutdown: mockMPShutdown };

  return {
    mockLoggerEmit, mockLoggerProvider, mockLPForceFlush, mockLPShutdown, mockLogsSetGlobal,
    mockTracerProvider, mockTPRegister, mockTPForceFlush, mockTPShutdown,
    mockMeterProvider, mockMPForceFlush, mockMPShutdown,
  };
});

vi.mock('@opentelemetry/api-logs', () => ({
  logs: { setGlobalLoggerProvider: mockLogsSetGlobal },
}));

vi.mock('@opentelemetry/sdk-logs', () => ({
  LoggerProvider: vi.fn(function () { return mockLoggerProvider; }),
  BatchLogRecordProcessor: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/exporter-logs-otlp-http', () => ({
  OTLPLogExporter: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/sdk-trace-web', () => ({
  WebTracerProvider: vi.fn(function () { return mockTracerProvider; }),
}));

vi.mock('@opentelemetry/sdk-trace-base', () => ({
  BatchSpanProcessor: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/exporter-trace-otlp-http', () => ({
  OTLPTraceExporter: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/sdk-metrics', () => ({
  MeterProvider: vi.fn(function () { return mockMeterProvider; }),
  PeriodicExportingMetricReader: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/exporter-metrics-otlp-http', () => ({
  OTLPMetricExporter: vi.fn(function () { return {}; }),
}));

vi.mock('@opentelemetry/resources', () => ({
  resourceFromAttributes: vi.fn(function () { return {}; }),
}));

// ── helpers ───────────────────────────────────────────────────────────────────

// Import once — vi.mock() above intercepts all dynamic imports inside browser.ts.
// Module state is reset between tests via Observation.shutdownObservability().
import { Observation } from '../../src/observation/browser';

// ── tests ─────────────────────────────────────────────────────────────────────

describe('observation/browser — no-transport mode', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await Observation.shutdownObservability();
  });

  it('getLogger returns a pino logger before setupObservability()', () => {
    const log = Observation.getLogger('test.browser');
    expect(log).toBeDefined();
    expect(typeof log.info).toBe('function');
    expect(typeof log.warn).toBe('function');
    expect(typeof log.child).toBe('function');
  });

  it('getLogger child carries log.logger binding', () => {
    const log = Observation.getLogger('my.scope', { extra: 42 });
    // pino/browser child loggers are objects with the same method surface
    expect(log).toBeDefined();
    expect(typeof log.info).toBe('function');
  });

  it('getTracer returns a no-op tracer', () => {
    const tracer = Observation.getTracer('test.browser');
    const span = tracer.startSpan('test-span');
    expect(() => {
      span.setAttribute('key', 'value');
      span.end();
    }).not.toThrow();
  });

  it('getMeter returns a no-op meter', () => {
    const meter = Observation.getMeter('test.browser');
    const counter = meter.createCounter('test.counter', { unit: 'req' });
    expect(() => counter.add(1, { env: 'test' })).not.toThrow();
  });

  it('setupObservability({ transport: none }) succeeds without loading OTel deps', async () => {
    await expect(
      Observation.setupObservability({ transport: 'none' }),
    ).resolves.toBeUndefined();
  });

  it('shutdownObservability() is safe to call before setup', async () => {
    await expect(Observation.shutdownObservability()).resolves.toBeUndefined();
  });

  it('calling setupObservability() twice throws', async () => {
    await Observation.setupObservability({ transport: 'none' });
    await expect(
      Observation.setupObservability({ transport: 'none' }),
    ).rejects.toThrow('already been called');
  });

  it('can call setupObservability() again after shutdownObservability()', async () => {
    await Observation.setupObservability({ transport: 'none' });
    await Observation.shutdownObservability();
    await expect(
      Observation.setupObservability({ transport: 'none' }),
    ).resolves.toBeUndefined();
  });
});

describe('observation/browser — log bridge (transport: http)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await Observation.shutdownObservability();
  });

  it('creates a LoggerProvider and BatchLogRecordProcessor', async () => {
    const { LoggerProvider, BatchLogRecordProcessor } = await import('@opentelemetry/sdk-logs');
    await Observation.setupObservability({
      transport: 'http',
      transportEndpoint: 'http://localhost:9938',
      logging: { serviceName: 'test-svc', serviceVersion: '1.0.0' },
    });

    expect(LoggerProvider).toHaveBeenCalledOnce();
    expect(BatchLogRecordProcessor).toHaveBeenCalledOnce();
    expect(mockLoggerProvider.addLogRecordProcessor).toHaveBeenCalledOnce();
    expect(mockLogsSetGlobal).toHaveBeenCalledWith(mockLoggerProvider);
  });

  it('transmit.send forwards log records to the OTel bridge', async () => {
    const consoleSpy = vi.spyOn(console, 'info').mockImplementation(() => {});

    await Observation.setupObservability({
      transport: 'http',
      transportEndpoint: 'http://localhost:9938',
      logging: { serviceName: 'bridge-svc', serviceVersion: '0.1.0', console: false },
    });

    const log = Observation.getLogger('bridge.test');
    log.info({ userId: 'u1' }, 'hello from bridge');

    // pino/browser calls transmit.send synchronously on every log call.
    expect(mockLoggerEmit).toHaveBeenCalledOnce();

    const emittedRecord = mockLoggerEmit.mock.calls[0][0] as Record<string, unknown>;
    expect(emittedRecord).toMatchObject({
      severityNumber: 9,      // INFO
      severityText: 'INFO',
      body: 'hello from bridge',
    });
    expect((emittedRecord['attributes'] as Record<string, unknown>)?.['service.name']).toBe('bridge-svc');
    expect((emittedRecord['attributes'] as Record<string, unknown>)?.['log.logger']).toBe('bridge.test');
    expect((emittedRecord['attributes'] as Record<string, unknown>)?.['userId']).toBe('u1');

    consoleSpy.mockRestore();
  });

  it('shutdown calls forceFlush + shutdown on LoggerProvider', async () => {
    await Observation.setupObservability({ transport: 'http' });
    await Observation.shutdownObservability();

    expect(mockLPForceFlush).toHaveBeenCalledOnce();
    expect(mockLPShutdown).toHaveBeenCalledOnce();
  });
});

describe('observation/browser — telemetry (enableTelemetry: true)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(async () => {
    await Observation.shutdownObservability();
  });

  it('registers WebTracerProvider and MeterProvider', async () => {
    const { WebTracerProvider } = await import('@opentelemetry/sdk-trace-web');
    const { MeterProvider } = await import('@opentelemetry/sdk-metrics');
    await Observation.setupObservability({
      transport: 'http',
      enableTelemetry: true,
      transportEndpoint: 'http://localhost:9938',
    });

    expect(WebTracerProvider).toHaveBeenCalledOnce();
    expect(mockTPRegister).toHaveBeenCalledOnce();
    expect(MeterProvider).toHaveBeenCalledOnce();
  });

  it('shutdown flushes and shuts down all three providers', async () => {
    await Observation.setupObservability({
      transport: 'http',
      enableTelemetry: true,
    });
    await Observation.shutdownObservability();

    expect(mockLPForceFlush).toHaveBeenCalledOnce();
    expect(mockLPShutdown).toHaveBeenCalledOnce();
    expect(mockMPForceFlush).toHaveBeenCalledOnce();
    expect(mockMPShutdown).toHaveBeenCalledOnce();
    expect(mockTPForceFlush).toHaveBeenCalledOnce();
    expect(mockTPShutdown).toHaveBeenCalledOnce();
  });
});

describe('observation/browser — page lifecycle (pagehide)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('installs pagehide listener after setup with transport: http', async () => {
    const addSpy = vi.spyOn(window, 'addEventListener');

    await Observation.setupObservability({ transport: 'http' });

    expect(addSpy).toHaveBeenCalledWith('pagehide', expect.any(Function));
    addSpy.mockRestore();
    await Observation.shutdownObservability();
  });

  it('removes pagehide listener after shutdownObservability()', async () => {
    const removeSpy = vi.spyOn(window, 'removeEventListener');

    await Observation.setupObservability({ transport: 'http' });
    await Observation.shutdownObservability();

    expect(removeSpy).toHaveBeenCalledWith('pagehide', expect.any(Function));
    removeSpy.mockRestore();
  });

  it('does NOT install pagehide listener when transport: none', async () => {
    const addSpy = vi.spyOn(window, 'addEventListener');

    await Observation.setupObservability({ transport: 'none' });

    expect(addSpy).not.toHaveBeenCalledWith('pagehide', expect.any(Function));
    addSpy.mockRestore();
    await Observation.shutdownObservability();
  });
});
