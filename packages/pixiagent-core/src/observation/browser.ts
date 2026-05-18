/**
 * Browser-compatible observation module — structured logging, distributed tracing, and metrics.
 *
 * Bundlers (Vite, webpack, esbuild) resolve this file automatically via the `browser`
 * condition in `package.json` exports whenever `@pixiagent/core/observation` is imported
 * in a browser bundle.  The Node.js implementation (`observation.ts`) is used otherwise.
 *
 * ## Logging
 *
 * Uses `pino/browser` (resolved by the bundler from `pino`'s own `browser` field).
 * Pino's `.info()` / `.warn()` / `.child()` API is fully preserved.
 * When `transport !== 'none'`, a `transmit` callback bridges every log record to the
 * OTel Logs SDK, which batches and exports them via OTLP/HTTP (fetch).
 *
 * ## Tracing / Metrics
 *
 * Requires `enableTelemetry: true`.  Uses `WebTracerProvider` + `MeterProvider` with
 * OTLP/HTTP exporters (gRPC is not available in browsers).
 *
 * ## Optional peer deps
 *
 * Install only the groups you need:
 *
 *   Log export (`transport !== 'none'`):
 *     bun add @opentelemetry/api-logs @opentelemetry/sdk-logs @opentelemetry/exporter-logs-otlp-http
 *
 *   Tracing + Metrics (`enableTelemetry: true`):
 *     bun add @opentelemetry/sdk-trace-web @opentelemetry/sdk-trace-base
 *             @opentelemetry/sdk-metrics
 *             @opentelemetry/exporter-trace-otlp-http
 *             @opentelemetry/exporter-metrics-otlp-http
 *             @opentelemetry/resources
 */

import pino from 'pino/browser';
import type { Logger as PinoLogger, LogEvent, LoggerOptions } from 'pino';
import { trace, metrics, context, propagation } from '@opentelemetry/api';

// ── types ─────────────────────────────────────────────────────────────────────

/**
 * Logging configuration — two variants supported in browser:
 *
 * - `{ rootLogger }` — use this pre-built pino logger directly.
 * - `{ serviceName?, serviceVersion?, console? }` — build a default logger
 *   (this is the default when `logging` is omitted).
 */
export type LoggingConfig =
  | { rootLogger: PinoLogger }
  | { serviceName?: string; serviceVersion?: string; console?: boolean };

export interface ObservabilityOptions {
  /**
   * Enable distributed tracing and metrics via WebTracerProvider + MeterProvider.
   * When false (default), only logging is set up.
   * @default false
   */
  enableTelemetry?: boolean;
  /**
   * OTLP exporter transport protocol.
   * - 'http': OTLP/HTTP via fetch (default endpoint: `http://localhost:9938`)
   * - 'none': no export to OTel collector; logs go to console only, traces/metrics are no-op
   *
   * Note: gRPC is not available in browsers.
   */
  transport: 'http' | 'none';
  /**
   * OTLP collector endpoint (HTTP base URL, e.g. `'http://localhost:9938'`).
   * Ignored when `transport` is `'none'`.
   * @default 'http://localhost:9938'
   */
  transportEndpoint?: string;
  /**
   * How often metrics are exported to the collector (ms).
   * @default 30_000
   */
  metricExportIntervalMs?: number;
  /**
   * Logging configuration.
   * Defaults to console output with service name `'pixiagent'` when omitted.
   */
  logging?: LoggingConfig;
}

// ── OTel type-only imports (erased at compile time) ───────────────────────────

import type { LoggerProvider } from '@opentelemetry/sdk-logs';
import type { WebTracerProvider } from '@opentelemetry/sdk-trace-web';
import type { MeterProvider } from '@opentelemetry/sdk-metrics';

// ── module state ──────────────────────────────────────────────────────────────

const DEFAULT_SERVICE_NAME = 'pixiagent';
const DEFAULT_SERVICE_VERSION = 'unknown';

let _rootLogger: PinoLogger | undefined;
let _tracerProvider: WebTracerProvider | undefined;
let _meterProvider: MeterProvider | undefined;
let _loggerProvider: LoggerProvider | undefined;
// OTel log bridge — set during setupObservability(); referenced by transmit.send closure.
let _otelLogBridge: { emit: (record: Record<string, unknown>) => void } | undefined;
let _isSetUp = false;
let _shutdownPromise: Promise<void> | undefined;

// ── internal: pino level → OTel SeverityNumber ───────────────────────────────
// Raw numbers per the OTel Logs Data Model spec (TRACE=1, DEBUG=5, INFO=9,
// WARN=13, ERROR=17, FATAL=21) — avoids a runtime import of @opentelemetry/api-logs.

function levelToSeverityNumber(level: string): number {
  switch (level) {
    case 'trace': return 1;
    case 'debug': return 5;
    case 'info':  return 9;
    case 'warn':  return 13;
    case 'error': return 17;
    case 'fatal': return 21;
    default:      return 0;
  }
}

// ── internal: logger building ─────────────────────────────────────────────────

/**
 * The transmit.send callback is a closure that references `_otelLogBridge`.
 * It is called synchronously by pino/browser on every log record.
 * Because setupObservability() is async, logs emitted before the first await
 * in setupObservability() will have _otelLogBridge === undefined and will be
 * dropped from OTel export (but still printed to console).  Logs emitted after
 * `await setupObservability()` returns are always captured.
 */
function makeTransmitSend(serviceName: string, serviceVersion: string) {
  return (level: string, logEvent: LogEvent) => {
    if (!_otelLogBridge) return;

    const attrs: Record<string, unknown> = {
      'service.name': serviceName,
      'service.version': serviceVersion,
    };

    // Merge accumulated child bindings (e.g. { 'log.logger': 'foo', runId: 'x' }).
    for (const b of logEvent.bindings) {
      Object.assign(attrs, b);
    }

    // Scan messages: strings become the body, objects are merged as attributes,
    // Error instances are serialised into exception.* attributes.
    let body = '';
    for (const m of logEvent.messages) {
      if (typeof m === 'string') {
        body = body ? `${body} ${m}` : m;
      } else if (m instanceof Error) {
        attrs['exception.type'] = m.name;
        attrs['exception.message'] = m.message;
        if (m.stack) attrs['exception.stacktrace'] = m.stack;
      } else if (m !== null && typeof m === 'object') {
        Object.assign(attrs, m as Record<string, unknown>);
      }
    }

    _otelLogBridge.emit({
      severityNumber: levelToSeverityNumber(level),
      severityText: level.toUpperCase(),
      body,
      attributes: attrs,
      timestamp: logEvent.ts,
    });
  };
}

function buildBrowserLogger(
  serviceName: string,
  serviceVersion: string,
  consoleOutput: boolean,
  shouldExportLogs: boolean,
): PinoLogger {
  const options: LoggerOptions = {
    level: 'info',
    browser: {
      // transmit fires on every log record, regardless of the write option.
      transmit: shouldExportLogs
        ? { level: 'trace', send: makeTransmitSend(serviceName, serviceVersion) }
        : undefined,
      // Override write with no-ops to suppress console output when console=false.
      write: consoleOutput
        ? undefined
        : {
            fatal: () => { /* suppress */ },
            error: () => { /* suppress */ },
            warn:  () => { /* suppress */ },
            info:  () => { /* suppress */ },
            debug: () => { /* suppress */ },
            trace: () => { /* suppress */ },
          },
    },
  };
  return pino(options);
}

// ── internal: page lifecycle handlers (replaces SIGTERM/SIGINT) ───────────────

function onPageHide(): void {
  void shutdownObservability();
}

function installUnloadHandlers(): void {
  if (typeof window !== 'undefined') {
    window.addEventListener('pagehide', onPageHide);
  }
}

function removeUnloadHandlers(): void {
  if (typeof window !== 'undefined') {
    window.removeEventListener('pagehide', onPageHide);
  }
}

// ── public API ────────────────────────────────────────────────────────────────

/**
 * Returns a child pino logger scoped to `name`.
 * Falls back to a default console logger when `setupObservability()` has not been called.
 */
function getLogger(name: string, bindings?: Record<string, unknown>): PinoLogger {
  if (!_rootLogger) {
    _rootLogger = buildBrowserLogger(DEFAULT_SERVICE_NAME, DEFAULT_SERVICE_VERSION, true, false);
  }
  return _rootLogger.child({ 'log.logger': name, ...bindings });
}

/**
 * Returns an OTel Tracer for the given instrumentation scope.
 * Returns a no-op tracer when `enableTelemetry` is false.
 */
function getTracer(name: string, version?: string) {
  return trace.getTracer(name, version);
}

/**
 * Returns an OTel Meter for the given instrumentation scope.
 * Returns a no-op meter when `enableTelemetry` is false.
 */
function getMeter(name: string, version?: string) {
  return metrics.getMeter(name, version);
}

/**
 * Initialises logging (and optionally the OTel WebTracerProvider + MeterProvider).
 *
 * All OTel SDK packages are loaded dynamically — they are not required when
 * `transport` is `'none'` and `enableTelemetry` is false.
 *
 * Throws if called more than once without an intervening `shutdownObservability()`.
 */
async function setupObservability(options: ObservabilityOptions): Promise<void> {
  if (_shutdownPromise) {
    throw new Error('[observation] setupObservability() cannot run while shutdown is in progress.');
  }
  if (_isSetUp) {
    throw new Error('[observation] setupObservability() has already been called.');
  }

  const {
    transport,
    transportEndpoint,
    enableTelemetry = false,
    metricExportIntervalMs = 30_000,
    logging,
  } = options;

  const resolvedEndpoint = transportEndpoint ?? 'http://localhost:9938';

  let serviceName = DEFAULT_SERVICE_NAME;
  let serviceVersion = DEFAULT_SERVICE_VERSION;
  if (logging && !('rootLogger' in logging)) {
    serviceName = logging.serviceName ?? DEFAULT_SERVICE_NAME;
    serviceVersion = logging.serviceVersion ?? DEFAULT_SERVICE_VERSION;
  }

  const shouldExportLogs = transport !== 'none';
  const consoleOutput =
    logging && !('rootLogger' in logging) ? (logging.console ?? true) : true;

  // ── OTel log bridge ─────────────────────────────────────────────────────────
  // Must be set up BEFORE building the pino logger so that the transmit.send
  // closure can reference _otelLogBridge after this function returns.
  if (shouldExportLogs) {
    const [
      { logs },
      { LoggerProvider, BatchLogRecordProcessor },
      { OTLPLogExporter },
    ] = await Promise.all([
      import('@opentelemetry/api-logs'),
      import('@opentelemetry/sdk-logs'),
      import('@opentelemetry/exporter-logs-otlp-http'),
    ]);

    const logExporter = new OTLPLogExporter({ url: `${resolvedEndpoint}/v1/logs` });
    const lp = new LoggerProvider();
    lp.addLogRecordProcessor(new BatchLogRecordProcessor(logExporter));
    logs.setGlobalLoggerProvider(lp);
    _loggerProvider = lp;
    // The bridge is referenced by the transmit.send closure (makeTransmitSend).
    _otelLogBridge = lp.getLogger(serviceName) as unknown as { emit: (r: Record<string, unknown>) => void };
  }

  // ── Tracing + Metrics (optional) ────────────────────────────────────────────
  if (enableTelemetry && transport !== 'none') {
    const [
      { WebTracerProvider },
      { BatchSpanProcessor },
      { OTLPTraceExporter },
      { MeterProvider, PeriodicExportingMetricReader },
      { OTLPMetricExporter },
      { resourceFromAttributes },
    ] = await Promise.all([
      import('@opentelemetry/sdk-trace-web'),
      import('@opentelemetry/sdk-trace-base'),
      import('@opentelemetry/exporter-trace-otlp-http'),
      import('@opentelemetry/sdk-metrics'),
      import('@opentelemetry/exporter-metrics-otlp-http'),
      import('@opentelemetry/resources'),
    ]);

    const resource = resourceFromAttributes({
      'service.name': serviceName,
      'service.version': serviceVersion,
    });

    const tp = new WebTracerProvider({
      resource,
      spanProcessors: [
        new BatchSpanProcessor(
          new OTLPTraceExporter({ url: `${resolvedEndpoint}/v1/traces` }),
        ),
      ],
    });
    tp.register();
    _tracerProvider = tp;

    const mp = new MeterProvider({
      resource,
      readers: [
        new PeriodicExportingMetricReader({
          exporter: new OTLPMetricExporter({ url: `${resolvedEndpoint}/v1/metrics` }),
          exportIntervalMillis: metricExportIntervalMs,
        }),
      ],
    });
    metrics.setGlobalMeterProvider(mp);
    _meterProvider = mp;
  }

  // ── Root logger ─────────────────────────────────────────────────────────────
  if (logging && 'rootLogger' in logging) {
    _rootLogger = logging.rootLogger;
  } else {
    _rootLogger = buildBrowserLogger(serviceName, serviceVersion, consoleOutput, shouldExportLogs);
  }

  _isSetUp = true;

  if (_loggerProvider || _tracerProvider) {
    installUnloadHandlers();
  }
}

/**
 * Gracefully shuts down all OTel providers, flushing pending telemetry.
 * Resets all module state so `setupObservability()` can be called again.
 */
async function shutdownObservability(): Promise<void> {
  if (_shutdownPromise) {
    return _shutdownPromise;
  }

  _shutdownPromise = (async () => {
    removeUnloadHandlers();

    const errors: string[] = [];

    if (_loggerProvider) {
      try {
        await _loggerProvider.forceFlush();
        await _loggerProvider.shutdown();
      } catch (err) {
        errors.push(`logger: ${String(err)}`);
      }
    }

    if (_meterProvider) {
      try {
        await _meterProvider.forceFlush();
        await _meterProvider.shutdown();
      } catch (err) {
        errors.push(`meter: ${String(err)}`);
      }
    }

    if (_tracerProvider) {
      try {
        await _tracerProvider.forceFlush();
        await _tracerProvider.shutdown();
      } catch (err) {
        errors.push(`tracer: ${String(err)}`);
      }
    }

    // Reset global OTel providers.
    try { metrics.disable(); } catch { /* ignore */ }
    try { trace.disable(); } catch { /* ignore */ }
    try { propagation.disable(); } catch { /* ignore */ }
    try { context.disable(); } catch { /* ignore */ }

    if (errors.length > 0) {
      console.error(`[observation] shutdown errors: ${errors.join(', ')}`);
    }

    _rootLogger = undefined;
    _tracerProvider = undefined;
    _meterProvider = undefined;
    _loggerProvider = undefined;
    _otelLogBridge = undefined;
    _isSetUp = false;
  })();

  try {
    await _shutdownPromise;
  } finally {
    _shutdownPromise = undefined;
  }
}

// ── namespace ─────────────────────────────────────────────────────────────────

const _getLogger = getLogger;
const _getTracer = getTracer;
const _getMeter = getMeter;
const _setupObservability = setupObservability;
const _shutdownObservability = shutdownObservability;
type _ObservabilityOptions = ObservabilityOptions;
type _LoggingConfig = LoggingConfig;

// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace Observation {
  export const getLogger = _getLogger;
  export const getTracer = _getTracer;
  export const getMeter = _getMeter;
  export const setupObservability = _setupObservability;
  export const shutdownObservability = _shutdownObservability;
  export type ObservabilityOptions = _ObservabilityOptions;
  export type LoggingConfig = _LoggingConfig;
}

// ── convenience re-exports ────────────────────────────────────────────────────
export {
  _getLogger as getLogger,
  _getTracer as getTracer,
  _getMeter as getMeter,
  _setupObservability as setupObservability,
  _shutdownObservability as shutdownObservability,
};
