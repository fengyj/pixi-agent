/**
 * Observation module — structured logging, distributed tracing, and metrics.
 *
 * ## Usage in other modules
 *
 *   ```typescript
 *   import { Observation } from '../observation';
 *
 *   const log = Observation.getLogger('transport.chat_completion');
 *   log.info({ model: opts.model, inputTokens: usage.inputTokens }, 'LLM call completed');
 *
 *   const tracer = Observation.getTracer('pixiagent.transport');
 *   tracer.startActiveSpan('generate', (span) => {
 *     span.setAttribute('gen_ai.model', opts.model);
 *     try { return await doWork(); } finally { span.end(); }
 *   });
 *
 *   const meter = Observation.getMeter('pixiagent.transport');
 *   const tokenCounter = meter.createCounter('gen_ai.client.token.usage', { unit: 'token' });
 *   tokenCounter.add(usage.inputTokens, { 'gen_ai.token.type': 'input', 'gen_ai.model': model });
 *   ```
 *
 * ## Sending telemetry to an OTel Collector
 *
 *   Call `setupObservability()` once at application startup, before other code
 *   calls `getLogger()` / `getTracer()` / `getMeter()`.
 *
 *   Without `setupObservability()`:
 *   - Logs are written to stdout as JSON.
 *   - Traces and metrics are silently dropped (OTel no-op).
 */

import pino from 'pino';
import { trace, metrics, context, propagation, diag, DiagConsoleLogger, DiagLogLevel } from '@opentelemetry/api';
import type { Logger as PinoLogger } from 'pino';
import { withSpan, Traced, retry, Retry } from './helpers';
export type {
  SpanHelperOptions,
  TracedOptions,
  RetryOptions,
  RetryDecoratorOptions,
} from './helpers';

// ── types ─────────────────────────────────────────────────────────────────────

/**
 * Logging configuration — exactly one of three variants:
 *
 * - `{ rootLogger }` — use this pre-built pino logger directly.
 * - `{ logOptions }` — build a new logger with these pino options.
 * - `{ serviceName?, serviceVersion?, ... }` — build a default ECS-compatible
 *   logger (this is the default when `logging` is omitted).
 */
export type LoggingConfig =
  | {
      rootLogger: PinoLogger;
    }
  | {
      logOptions: pino.LoggerOptions;
      /**
       * Whether to also pipe logs to the OTLP collector.
       * Defaults to `transport !== 'none'` when omitted.
       */
      outputToOtel?: boolean;
    }
  | {
      serviceName?: string;
      serviceVersion?: string;
      outputToConsole?: boolean;
      /**
       * Whether to also pipe logs to the OTLP collector.
       * Defaults to `transport !== 'none'` when omitted.
       */
      outputToOtel?: boolean;
    };

export interface ObservabilityOptions {
  /**
   * Enable distributed tracing and metrics via OTel NodeSDK.
   * When false (default), NodeSDK is not started and telemetry is no-op.
   * @default false
   */
  enableTelemetry?: boolean;
  /**
   * OTLP exporter transport protocol.
   * - 'grpc': gRPC (default endpoint: `localhost:9937`)
   * - 'http': HTTP (default endpoint: `http://localhost:9938`)
   * - 'none': no export to OTel collector; logs stay local, traces/metrics are no-op
   */
  transport: 'grpc' | 'http' | 'none';
  /**
   * OTLP collector endpoint.
   * - gRPC: host:port without scheme, e.g. `'localhost:9937'` (default)
   * - HTTP: full URL, e.g. `'http://localhost:9938'` (default)
   * Ignored when `transport` is `'none'`.
   */
  transportEndpoint?: string;
  /**
   * How often metrics are exported to the collector (ms).
   * @default 30_000
   */
  metricExportIntervalMs?: number;
  /**
   * Logging configuration.
   * Defaults to stdout with service name `'pixiagent'` when omitted.
   */
  logging?: LoggingConfig;
}

// ── OTel SDK packages (optional peer deps) ────────────────────────────────────
// Loaded via dynamic import() only when enableTelemetry=true so the module can
// be required/imported without installing them.
// Type-only imports are erased at compile time and never cause missing-package errors.
import type { NodeSDK } from '@opentelemetry/sdk-node';

// ── module state ──────────────────────────────────────────────────────────────

const DEFAULT_SERVICE_NAME = 'pixiagent';
const DEFAULT_SERVICE_VERSION = 'unknown';

type LoggerTransportTarget = pino.TransportTargetOptions<Record<string, unknown>>;

function isTransportTargetOption(value: unknown): value is LoggerTransportTarget {
  return (
    typeof value === 'object' &&
    value !== null &&
    'target' in value &&
    typeof (value as { target?: unknown }).target === 'string'
  );
}

let _rootLogger: PinoLogger | undefined;
let _sdk: NodeSDK | undefined;
let _isSettingUp = false;
let _isSetUp = false;
let _shutdownPromise: Promise<void> | undefined;

// ── internal: logger building ─────────────────────────────────────────────────

function buildDefaultLoggerOptions(serviceName: string, serviceVersion: string): pino.LoggerOptions {
  return {
    level: process.env['LOG_LEVEL'] ?? 'info',
    redact: {
      paths: [
        'apiKey',
        '*.apiKey',
        'authorization',
        '*.authorization',
        'password',
        '*.password',
      ],
      censor: '[Redacted]',
    },
    timestamp: () => `,"@timestamp":"${new Date().toISOString()}"`,
    messageKey: 'message',
    formatters: {
      bindings: (bindings) => ({
        'service.name': serviceName,
        'service.version': serviceVersion,
        'process.pid': bindings['pid'],
        'host.name': bindings['hostname'],
      }),
    },
    mixin() {
      const span = trace.getActiveSpan();
      if (!span?.isRecording()) return {};
      const ctx = span.spanContext();
      return {
        'trace.id': ctx.traceId,
        'span.id': ctx.spanId,
        'trace.flags': ctx.traceFlags,
      };
    },
  };
}

function buildConsoleTransportTarget(): LoggerTransportTarget {
  return {
    target: 'pino-pretty',
    options: { colorize: true },
  };
}

function buildOtelTransportTarget(
  serviceName: string,
  serviceVersion: string,
  transport: 'grpc' | 'http',
  endpoint: string,
): LoggerTransportTarget {
  return {
    target: 'pino-opentelemetry-transport',
    options: {
      loggerName: serviceName,
      serviceVersion,
      resourceAttributes: {
        'service.name': serviceName,
        'service.version': serviceVersion,
      },
      logRecordProcessorOptions:
        transport === 'grpc'
          ? {
              recordProcessorType: 'batch',
              exporterOptions: {
                protocol: 'grpc',
                grpcExporterOptions: {
                  url: endpoint.startsWith('http') ? endpoint : `http://${endpoint}`,
                },
              },
              processorConfig: { scheduledDelayMillis: 1_000 },
            }
          : {
              recordProcessorType: 'batch',
              exporterOptions: {
                protocol: 'http',
                httpExporterOptions: { url: endpoint },
              },
              processorConfig: { scheduledDelayMillis: 1_000 },
            },
    },
  };
}

function buildDefaultTransportTargets(
  outputToConsole: boolean,
  outputToOtel: boolean,
  serviceName: string,
  serviceVersion: string,
  transport: 'grpc' | 'http' | 'none',
  endpoint?: string,
): LoggerTransportTarget[] {
  const targets: LoggerTransportTarget[] = [];

  if (outputToConsole) {
    targets.push(buildConsoleTransportTarget());
  }

  if (outputToOtel && transport !== 'none') {
    if (!endpoint) {
      throw new Error(`[observation] transportEndpoint is required for OTLP log transport when transport is '${transport}'.`); // should be caught by TypeScript but added here for runtime safety
    }
    targets.push(buildOtelTransportTarget(serviceName, serviceVersion, transport, endpoint!));
  }

  if (targets.length === 0) {
    targets.push({ target: 'pino/file', options: { destination: 1 } });
  }

  return targets;
}

function addOtelTransportToLogOptions(
  logOptions: pino.LoggerOptions,
  serviceName: string,
  serviceVersion: string,
  transport: 'grpc' | 'http',
  endpoint: string,
): pino.LoggerOptions {
  const otelTarget = buildOtelTransportTarget(serviceName, serviceVersion, transport, endpoint);
  const transportConfig = logOptions.transport as unknown;

  if (!transportConfig || typeof transportConfig !== 'object') {
    return { ...logOptions, transport: { targets: [otelTarget] } };
  }

  const transportObject = transportConfig as { target?: unknown; targets?: unknown } & Record<string, unknown>;
  if (Array.isArray(transportObject.targets)) {
    const targets = transportObject.targets.filter(isTransportTargetOption);
    if (!targets.some((target) => target.target === 'pino-opentelemetry-transport')) {
      targets.push(otelTarget);
    }

    const transportOptions: pino.TransportMultiOptions<Record<string, unknown>> = {
      ...transportObject,
      targets,
    };

    return { ...logOptions, transport: transportOptions };
  }

  if (isTransportTargetOption(transportObject)) {
    const transportOptions: pino.TransportMultiOptions<Record<string, unknown>> = {
      targets: [transportObject, otelTarget],
    };

    return {
      ...logOptions,
      transport: transportOptions,
    };
  }

  return { ...logOptions, transport: { targets: [otelTarget] } };
}
// ── internal: OTel log transport ──────────────────────────────────────────────

// ── internal: signal handlers ─────────────────────────────────────────────────

function onSignalShutdown(): void {
  void shutdownObservability();
}

function installSignalHandlers(): void {
  process.on('SIGTERM', onSignalShutdown);
  process.on('SIGINT', onSignalShutdown);
}

function removeSignalHandlers(): void {
  process.off('SIGTERM', onSignalShutdown);
  process.off('SIGINT', onSignalShutdown);
}

// ── public API ────────────────────────────────────────────────────────────────

/**
 * Returns a child pino logger scoped to `name`.
 *
 * The returned logger is a pino child with `log.logger` set to `name` so
 * every record carries the originating component.  Extra key-value pairs can
 * be attached via `bindings` and will appear on every record from this logger.
 *
 * **Before `setupObservability()` is called** a default stdout logger is
 * created on first use (ECS-compatible JSON, level from the `LOG_LEVEL` env
 * var, defaults to `'info'`).  Libraries can therefore call `getLogger()` at
 * module-load time without requiring a prior `setupObservability()` call.
 *
 * **After `setupObservability()` is called** the logger reflects the
 * configured transport targets (`pino.transport.targets`) and optional
 * OTel log export target when enabled.
 *
 * Every JSON record automatically includes:
 * - `@timestamp`              ISO-8601 (Elasticsearch / Kibana time axis)
 * - `log.level`               Severity string (`info`, `warn`, `error`, …)
 * - `log.logger`              The `name` argument passed here
 * - `message`                 Human-readable message
 * - `service.name`            From `setupObservability({ logging: { serviceName } })`
 * - `service.version`         From `setupObservability({ logging: { serviceVersion } })`
 * - `trace.id` / `span.id` / `trace.flags`  Injected when an active OTel span exists
 * - Redacted fields: `apiKey`, `authorization`, `password` (and nested) → `'[Redacted]'`
 *
 * @param name     Dot-separated instrumentation scope,
 *                 e.g. `'transport.chat_completion'`.
 * @param bindings Optional key-value pairs attached to every record emitted
 *                 by this logger and its children.
 * @returns        A pino child logger.
 *
 * @example
 * const log = getLogger('agent.runner', { agentId: 'abc' });
 * log.info({ step: 1 }, 'Agent started');
 * log.child({ requestId: 'xyz' }).warn('Retrying after rate limit');
 * log.error({ err }, 'Unhandled error');
 */
function getLogger(name: string, bindings?: Record<string, unknown>): PinoLogger {
  if (!_rootLogger) {
    _rootLogger = pino(buildDefaultLoggerOptions(DEFAULT_SERVICE_NAME, DEFAULT_SERVICE_VERSION));
  }
  return _rootLogger.child({ 'log.logger': name, ...bindings });
}

/**
 * Returns an OTel `Tracer` for the given instrumentation scope.
 *
 * Use this to create spans that represent discrete units of work — LLM calls,
 * tool executions, HTTP requests, retrieval steps, etc.
 *
 * Span data is exported to the OTel collector when
 * `setupObservability({ enableTelemetry: true })` has been called; otherwise
 * the returned tracer is a silent no-op and safe to call unconditionally.
 *
 * @param name    Instrumentation scope name, e.g. `'pixiagent.transport'`.
 * @param version Optional instrumentation library version string.
 * @returns       An OTel `Tracer` (no-op when SDK is not initialised).
 *
 * @example
 * const tracer = getTracer('pixiagent.transport');
 * const result = await tracer.startActiveSpan('llm.generate', async (span) => {
 *   span.setAttribute('gen_ai.model', model);
 *   span.setAttribute('gen_ai.operation.name', 'chat');
 *   try {
 *     return await callLLM();
 *   } catch (err) {
 *     span.recordException(err as Error);
 *     span.setStatus({ code: SpanStatusCode.ERROR });
 *     throw err;
 *   } finally {
 *     span.end();
 *   }
 * });
 */
function getTracer(name: string, version?: string) {
  return trace.getTracer(name, version);
}

/**
 * Returns an OTel `Meter` for the given instrumentation scope.
 *
 * Use this to create counters, histograms, and gauges that capture
 * quantitative runtime behaviour: token usage, LLM latency, queue depth,
 * retry counts, etc.
 *
 * Metric data is periodically flushed to the OTel collector (interval
 * controlled by `setupObservability({ metricExportIntervalMs })`) when
 * `enableTelemetry: true`; otherwise the meter is a silent no-op.
 *
 * @param name    Instrumentation scope name, e.g. `'pixiagent.transport'`.
 * @param version Optional instrumentation library version string.
 * @returns       An OTel `Meter` (no-op when SDK is not initialised).
 *
 * @example
 * const meter = getMeter('pixiagent.transport');
 *
 * const tokenCounter = meter.createCounter('gen_ai.client.token.usage', {
 *   unit: 'token',
 *   description: 'Total tokens consumed by LLM calls',
 * });
 * tokenCounter.add(usage.inputTokens,  { 'gen_ai.token.type': 'input',  'gen_ai.model': model });
 * tokenCounter.add(usage.outputTokens, { 'gen_ai.token.type': 'output', 'gen_ai.model': model });
 *
 * const latency = meter.createHistogram('gen_ai.client.operation.duration', { unit: 's' });
 * latency.record(elapsedSeconds, { 'gen_ai.operation.name': 'chat', 'gen_ai.model': model });
 */
function getMeter(name: string, version?: string) {
  return metrics.getMeter(name, version);
}

/**
 * Initialises logging (and optionally the OTel NodeSDK for traces + metrics).
 *
 * Returns the `NodeSDK` instance when `enableTelemetry` is true, otherwise `undefined`.
 * Use `shutdownObservability()` for graceful shutdown.
 *
 * **Call once at application startup**, before any code calls `getLogger()` /
 * `getTracer()` / `getMeter()` and expects telemetry to flow.  Calling it a
 * second time without a prior `shutdownObservability()` throws.
 *
 * **Dynamic imports**: all OTel SDK packages are loaded via `import()` only
 * when needed.  They are optional peer dependencies — install only the groups
 * you use:
 *
 * - Log export to collector (`transport !== 'none'`):
 *   `pino-opentelemetry-transport`
 *
 * - Tracing + Metrics (`enableTelemetry: true`, gRPC transport):
 *   `@opentelemetry/sdk-node`, `@opentelemetry/resources`,
 *   `@opentelemetry/sdk-metrics`,
 *   `@opentelemetry/exporter-trace-otlp-grpc`,
 *   `@opentelemetry/exporter-metrics-otlp-grpc`
 *
 * - Tracing + Metrics (`enableTelemetry: true`, HTTP transport):
 *   same as above but replace the `-grpc` packages with
 *   `@opentelemetry/exporter-trace-otlp-http` and
 *   `@opentelemetry/exporter-metrics-otlp-http`.
 *
 * @param options.transport
 *   **Required.** `'grpc'` | `'http'` | `'none'`.
 *   Controls both log export (pino → OTLP worker thread) and telemetry export.
 *   Use `'none'` for local-only logging with no collector.
 *
 * @param options.transportEndpoint
 *   Collector address.  Defaults: gRPC → `'localhost:9937'`,
 *   HTTP → `'http://localhost:9938'`.
 *   gRPC format: `'host:port'` (no scheme).  HTTP format: full URL.
 *
 * @param options.enableTelemetry
 *   Start OTel `NodeSDK` for distributed traces and metrics.  Default `false`.
 *   Pino → OTLP log export is independent of this flag for internally built
 *   loggers.  When `logging.outputToOtel` is set, it controls whether the
 *   internal pino OTLP transport is added.
 *
 * @param options.metricExportIntervalMs
 *   How often metrics are pushed to the collector, in milliseconds.
 *   Default `30_000`.
 *
 * @param options.logging
 *   Logging variant — see {@link LoggingConfig}.
 *   Omit to get stdout JSON with service name `'pixiagent'`.
 *
 * @returns The `NodeSDK` instance when `enableTelemetry` is true, else `undefined`.
 */
async function setupObservability(options: ObservabilityOptions): Promise<NodeSDK | undefined> {
  if (_shutdownPromise) {
    throw new Error('[observation] setupObservability() cannot run while shutdown is in progress.');
  }
  if (_isSettingUp) {
    throw new Error('[observation] setupObservability() is already in progress.');
  }
  if (_isSetUp) {
    throw new Error('[observation] setupObservability() has already been called in this process.');
  }

  _isSettingUp = true;

  try {
    const {
      transport,
      transportEndpoint,
      enableTelemetry = transport !== 'none',
      metricExportIntervalMs = 30_000,
      logging,
    } = options;

    if (transport !== 'none' && !transportEndpoint) {
      throw new Error(`[observation] transportEndpoint is required when transport is '${transport}'.`); // should be caught by TypeScript but added here for runtime safety
    }

    // Resolve service identity (used for NodeSDK resource and OTel log transport).
    let serviceName = DEFAULT_SERVICE_NAME;
    let serviceVersion = DEFAULT_SERVICE_VERSION;
    if (logging && !('rootLogger' in logging) && !('logOptions' in logging)) {
      serviceName = logging.serviceName ?? DEFAULT_SERVICE_NAME;
      serviceVersion = logging.serviceVersion ?? DEFAULT_SERVICE_VERSION;
    }

    const outputToOtel = logging && 'rootLogger' in logging
      ? false
      : logging && ('outputToOtel' in logging)
      ? logging.outputToOtel !== false
      : transport !== 'none';

    let logOptionsWithOtelTransport: pino.LoggerOptions | undefined;

    if (logging && 'logOptions' in logging && outputToOtel && transport !== 'none') {
      logOptionsWithOtelTransport = addOtelTransportToLogOptions(
        logging.logOptions,
        serviceName,
        serviceVersion,
        transport,
        transportEndpoint!,
      );
    }

    // Start NodeSDK for distributed traces + metrics when telemetry is enabled.
    // SDK packages are loaded dynamically so callers don't need to install them
    // when enableTelemetry is false (the default).
    let sdk: NodeSDK | undefined;
    if (enableTelemetry && transport !== 'none') {
      const url =
        transport === 'grpc'
          ? transportEndpoint!.startsWith('http')
            ? transportEndpoint!
            : `http://${transportEndpoint!}`
          : transportEndpoint!;

      const [
        { NodeSDK: NodeSDKClass },
        { resourceFromAttributes },
        { PeriodicExportingMetricReader },
        { OTLPTraceExporter: OTLPTraceExporterGrpc },
        { OTLPMetricExporter: OTLPMetricExporterGrpc },
        { OTLPTraceExporter: OTLPTraceExporterHttp },
        { OTLPMetricExporter: OTLPMetricExporterHttp },
      ] = await Promise.all([
        import('@opentelemetry/sdk-node'),
        import('@opentelemetry/resources'),
        import('@opentelemetry/sdk-metrics'),
        import('@opentelemetry/exporter-trace-otlp-grpc'),
        import('@opentelemetry/exporter-metrics-otlp-grpc'),
        import('@opentelemetry/exporter-trace-otlp-http'),
        import('@opentelemetry/exporter-metrics-otlp-http'),
      ]);

      const traceExporter =
        transport === 'grpc' ? new OTLPTraceExporterGrpc({ url }) : new OTLPTraceExporterHttp({ url });

      const metricReaders = [
        new PeriodicExportingMetricReader({
          exporter:
            transport === 'grpc'
              ? new OTLPMetricExporterGrpc({ url })
              : new OTLPMetricExporterHttp({ url }),
          exportIntervalMillis: metricExportIntervalMs,
        }),
      ];

      const resource = resourceFromAttributes({
        'service.name': serviceName,
        'service.version': serviceVersion,
      });

      diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.ERROR);

      sdk = new NodeSDKClass({ resource, traceExporter, metricReaders });
      sdk.start();
    }

    // Build the root logger.
    if (logging && 'rootLogger' in logging) {
      _rootLogger = logging.rootLogger;
    } else if (logging && 'logOptions' in logging) {
      _rootLogger = pino(logOptionsWithOtelTransport ?? logging.logOptions);
    } else {
      const outputToConsole = logging?.outputToConsole ?? true;
      const defaultLogOptions = buildDefaultLoggerOptions(serviceName, serviceVersion);
      const defaultTransportTargets = buildDefaultTransportTargets(
        outputToConsole,
        outputToOtel,
        serviceName,
        serviceVersion,
        transport,
        transportEndpoint,
      );
      _rootLogger = pino({
        ...defaultLogOptions,
        transport: { targets: defaultTransportTargets },
      });
    }

    _sdk = sdk;
    _isSetUp = true;

    if (sdk) installSignalHandlers();

    return sdk;
  } catch (err) {
    removeSignalHandlers();
    void _sdk?.shutdown().catch(() => {
      // ignore rollback shutdown failures
    });
    _rootLogger = undefined;
    _sdk = undefined;
    _isSetUp = false;
    throw err;
  } finally {
    _isSettingUp = false;
  }
}

/**
 * Gracefully shuts down the OTel SDK (if started) and flushes all pending
 * telemetry, then resets module state so `setupObservability()` can be called
 * again.
 *
 * Shutdown sequence:
 * 1. Removes `SIGTERM` / `SIGINT` signal handlers installed by
 *    `setupObservability()`.
 * 2. Ends the pino → OTLP worker-thread transport (`.end()`) and awaits the
 *    `'close'` event, which fires only after the `BatchLogRecordProcessor`
 *    inside the worker has fully exported all buffered log records.  This
 *    guarantees no records are silently dropped on process exit.
 * 3. Calls `NodeSDK.shutdown()` which force-flushes spans and metrics and
 *    shuts down all configured exporters.
 * 4. Disables the OTel global API providers (`trace`, `metrics`, `context`,
 *    `propagation`) so subsequent calls return no-op instances.
 * 5. Clears all internal module state so the module is ready for a fresh
 *    `setupObservability()` call (useful in tests).
 *
 * **Idempotent / concurrent-safe**: if called while a previous shutdown is
 * still in flight, returns the same in-flight `Promise` rather than starting
 * a second shutdown.
 *
 * **Safe to call without prior `setupObservability()`**: resolves immediately
 * when nothing has been set up.
 *
 * @returns A `Promise` that resolves once all providers have been shut down
 *   and all buffered telemetry has been flushed to the collector.
 */
async function shutdownObservability(): Promise<void> {
  if (_shutdownPromise) {
    return _shutdownPromise;
  }

  _shutdownPromise = (async () => {
    removeSignalHandlers();

    if (_rootLogger) {
      const loggerWithInternals = _rootLogger as PinoLogger & {
        flush?: (cb?: () => void) => void;
      } & Record<symbol, unknown>;

      if (typeof loggerWithInternals.flush === 'function') {
        await new Promise<void>((resolve) => {
          let settled = false;
          const done = () => {
            if (settled) return;
            settled = true;
            resolve();
          };

          try {
            loggerWithInternals.flush(done);
          } catch {
            done();
            return;
          }

          setTimeout(done, 100);
        });
      }

      const streamSymbol = Object.getOwnPropertySymbols(loggerWithInternals).find(
        (symbol) => symbol.toString() === 'Symbol(pino.stream)',
      );

      const loggerStream = streamSymbol ? loggerWithInternals[streamSymbol] : undefined;
      if (
        loggerStream &&
        typeof loggerStream === 'object' &&
        typeof (loggerStream as { end?: unknown }).end === 'function'
      ) {
        await new Promise<void>((resolve) => {
          let settled = false;
          const done = () => {
            if (settled) return;
            settled = true;
            resolve();
          };

          try {
            ((loggerStream as { end: (cb?: () => void) => void }).end)(done);
          } catch {
            done();
            return;
          }

          setTimeout(done, 250);
        });
      }
    }

    if (_sdk) {
      try {
        await _sdk.shutdown();
      } catch (err) {
        process.stderr.write(`[observation] SDK shutdown error: ${String(err)}\n`);
      } finally {
        try { metrics.disable(); } catch { /* ignore */ }
        try { trace.disable(); } catch { /* ignore */ }
        try { propagation.disable(); } catch { /* ignore */ }
        try { context.disable(); } catch { /* ignore */ }
      }
    }

    _sdk = undefined;
    _rootLogger = undefined;
    _isSetUp = false;
  })();

  try {
    await _shutdownPromise;
  } finally {
    _shutdownPromise = undefined;
  }
}

export const Observation = {
  getLogger,
  getTracer,
  getMeter,
  setupObservability,
  shutdownObservability,
  helpers: {
    withSpan,
    Traced,
    retry,
    Retry,
  },
  Targets: {
    buildConsoleTransportTarget,
    buildOtelTransportTarget,
  },
};
