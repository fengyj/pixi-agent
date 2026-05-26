import { SpanStatusCode, trace } from '@opentelemetry/api';
import type { Attributes, Exception, Span, Tracer } from '@opentelemetry/api';
import { isRetriableError } from '../errors/guards';

const DEFAULT_TRACER_NAME = 'pixiagent';

export interface SpanHelperOptions {
  tracer?: Tracer;
  attrs?: Attributes;
  isExpectedError?: (err: unknown) => boolean;
  /**
   * Whether to mark span status as OK when execution succeeds.
   * Defaults to false to avoid overriding status set by callback logic.
   */
  setOkStatus?: boolean;
}

export interface TracedOptions<TArgs extends unknown[], TResult> extends SpanHelperOptions {
  name?: string;
  attrsFn?: (...args: TArgs) => Attributes;
  resultAttrsFn?: (result: TResult) => Attributes;
}

export interface RetryOptions<T> extends SpanHelperOptions {
  span?: Span;
  maxAttempts?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  backoffMultiplier?: number;
  jitterRatio?: number;
  signal?: AbortSignal;
  shouldRetry?: (err: unknown, attempt: number, maxAttempts: number) => boolean;
  defaultResultOnFailure?: (err: unknown) => Promise<T>;
}

export interface RetryDecoratorOptions<
  TArgs extends unknown[],
  TResult,
> extends RetryOptions<TResult> {
  name?: string;
  attrsFn?: (...args: TArgs) => Attributes;
  resultAttrsFn?: (result: TResult) => Attributes;
  spanFromArgs?: (...args: TArgs) => Span | undefined;
}

function getTracerOrDefault(tracer?: Tracer): Tracer {
  return tracer ?? trace.getTracer(DEFAULT_TRACER_NAME);
}

function setAttributes(span: Span, attrs?: Attributes): void {
  if (!attrs) return;
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined) continue;
    span.setAttribute(key, value);
  }
}

function getErrorType(err: unknown): string {
  if (err instanceof Error) return err.name || 'Error';
  if (typeof err === 'object' && err !== null && 'name' in err) {
    const name = (err as { name?: unknown }).name;
    if (typeof name === 'string' && name.length > 0) return name;
  }
  return 'Error';
}

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  return String(err);
}

function toRecordableException(err: unknown): Exception {
  if (err instanceof Error || typeof err === 'string') return err;
  return {
    name: getErrorType(err),
    message: getErrorMessage(err),
  };
}

function handleSpanError(
  span: Span,
  err: unknown,
  isExpectedError?: (err: unknown) => boolean,
): void {
  const expected = isExpectedError?.(err) ?? false;
  span.recordException(toRecordableException(err));
  span.setAttribute('error.type', getErrorType(err));
  span.setAttribute('error.expected', expected);
  if (!expected) {
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: getErrorMessage(err),
    });
  }
}

function isPromiseLike<T>(value: unknown): value is PromiseLike<T> {
  return (
    typeof value === 'object' &&
    value !== null &&
    'then' in value &&
    typeof (value as { then?: unknown }).then === 'function'
  );
}

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const timeoutId: ReturnType<typeof setTimeout> = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);

    const onAbort = () => {
      clearTimeout(timeoutId);
      const reason = signal?.reason;
      reject(reason instanceof Error ? reason : new Error('Retry wait aborted.'));
    };

    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

function computeDelayMs<T>(opts: RetryOptions<T>, attempt: number): number {
  const initialDelayMs = Math.max(0, opts.initialDelayMs ?? 200);
  const maxDelayMs = Math.max(initialDelayMs, opts.maxDelayMs ?? 5_000);
  const backoffMultiplier = Math.max(1, opts.backoffMultiplier ?? 2);
  const jitterRatio = Math.max(0, Math.min(1, opts.jitterRatio ?? 0.15));

  const base = Math.min(maxDelayMs, initialDelayMs * Math.pow(backoffMultiplier, attempt - 1));
  if (jitterRatio === 0) return Math.round(base);
  const jitterRange = base * jitterRatio;
  const jittered = base - jitterRange + Math.random() * jitterRange * 2;
  return Math.max(0, Math.round(jittered));
}

async function executeWithRetrySpan<T>(
  span: Span,
  fn: (ctx: { span: Span; attempt: number }) => Promise<T> | T,
  opts: RetryOptions<T>,
): Promise<T> {
  const requestedAttempts = opts.maxAttempts;
  const maxAttempts =
    requestedAttempts === undefined || requestedAttempts <= 0
      ? 1
      : Math.max(1, Math.floor(requestedAttempts));
  setAttributes(span, opts.attrs);
  span.setAttribute('retry.max_attempts', maxAttempts);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    span.addEvent('retry.attempt.start', {
      'retry.attempt': attempt,
      'retry.max_attempts': maxAttempts,
    });

    try {
      const result = await fn({ span, attempt });
      span.setAttribute('retry.attempts', attempt);
      span.addEvent('retry.success', {
        'retry.attempt': attempt,
        'retry.max_attempts': maxAttempts,
      });
      if (opts.setOkStatus) {
        span.setStatus({ code: SpanStatusCode.OK });
      }
      return result;
    } catch (err) {
      const retriable = isRetriableError(err);
      const canRetry =
        retriable &&
        attempt < maxAttempts &&
        (opts.shouldRetry ? opts.shouldRetry(err, attempt, maxAttempts) : true);

      span.addEvent('retry.attempt.error', {
        'retry.attempt': attempt,
        'retry.max_attempts': maxAttempts,
        'retry.retriable': retriable,
        'retry.will_retry': canRetry,
        'error.type': getErrorType(err),
        'error.message': getErrorMessage(err),
      });

      if (!canRetry) {
        if (opts.defaultResultOnFailure !== undefined) {
          span.setAttribute('retry.give_up_with_default', true);
          if (opts.setOkStatus) {
            span.setStatus({ code: SpanStatusCode.OK });
          }
          return await opts.defaultResultOnFailure(err);
        } else {
          span.setAttribute('retry.give_up_at', attempt);
          handleSpanError(span, err, opts.isExpectedError);
          throw err;
        }
      }

      const delayMs = computeDelayMs(opts, attempt);
      span.addEvent('retry.backoff', {
        'retry.attempt': attempt,
        'retry.next_attempt': attempt + 1,
        'retry.delay_ms': delayMs,
      });

      await wait(delayMs, opts.signal);
    }
  }

  throw new Error('Retry exhausted unexpectedly.');
}

export async function withSpan<T>(
  name: string,
  fn: (span: Span) => Promise<T> | T,
  opts: SpanHelperOptions = {},
): Promise<T> {
  const tracer = getTracerOrDefault(opts.tracer);
  return tracer.startActiveSpan(name, async (span) => {
    setAttributes(span, opts.attrs);
    try {
      const result = await fn(span);
      if (opts.setOkStatus) span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (err) {
      handleSpanError(span, err, opts.isExpectedError);
      throw err;
    } finally {
      span.end();
    }
  });
}

export async function retry<T>(
  name: string,
  fn: (ctx: { span: Span; attempt: number }) => Promise<T> | T,
  opts: RetryOptions<T> = {},
): Promise<T> {
  if (opts.span) {
    return executeWithRetrySpan(opts.span, fn, opts);
  }

  const tracer = getTracerOrDefault(opts.tracer);
  return tracer.startActiveSpan(name, async (span) => {
    try {
      return await executeWithRetrySpan(span, fn, opts);
    } finally {
      span.end();
    }
  });
}

export function Retry<TArgs extends unknown[], TResult = unknown>(
  opts: RetryDecoratorOptions<TArgs, TResult> = {},
): MethodDecorator {
  return function (_target: object, key: string | symbol, descriptor: PropertyDescriptor): void {
    if (typeof descriptor.value !== 'function') {
      throw new TypeError('@Retry can only be applied to methods.');
    }

    const original = descriptor.value as (
      this: unknown,
      ...args: TArgs
    ) => TResult | Promise<TResult>;

    const wrapped = function (this: unknown, ...args: TArgs): Promise<TResult> {
      const spanFromArgs = opts.spanFromArgs?.(...args);
      const inheritedSpan = spanFromArgs ?? trace.getActiveSpan() ?? undefined;

      return retry<TResult>(
        opts.name ?? String(key),
        async ({ span }) => {
          const result = await original.apply(this, args);
          setAttributes(span, opts.resultAttrsFn?.(result));
          return result;
        },
        {
          span: inheritedSpan,
          tracer: opts.tracer,
          attrs: {
            ...(opts.attrs ?? {}),
            ...(opts.attrsFn?.(...args) ?? {}),
          },
          isExpectedError: opts.isExpectedError,
          setOkStatus: opts.setOkStatus,
          maxAttempts: opts.maxAttempts,
          initialDelayMs: opts.initialDelayMs,
          maxDelayMs: opts.maxDelayMs,
          backoffMultiplier: opts.backoffMultiplier,
          jitterRatio: opts.jitterRatio,
          signal: opts.signal,
          shouldRetry: opts.shouldRetry,
        },
      );
    };

    descriptor.value = wrapped as (...args: TArgs) => TResult | Promise<TResult>;
  } as MethodDecorator;
}

export function Traced<TArgs extends unknown[], TResult = unknown>(
  opts: TracedOptions<TArgs, TResult> = {},
): MethodDecorator {
  return function (_target: object, key: string | symbol, descriptor: PropertyDescriptor): void {
    if (typeof descriptor.value !== 'function') {
      throw new TypeError('@Traced can only be applied to methods.');
    }

    const original = descriptor.value as (
      this: unknown,
      ...args: TArgs
    ) => TResult | Promise<TResult>;

    const wrapped = function (this: unknown, ...args: TArgs): TResult | Promise<TResult> {
      const spanName = opts.name ?? String(key);
      const tracer = getTracerOrDefault(opts.tracer);

      return tracer.startActiveSpan(spanName, (span) => {
        let isAsync = false;
        setAttributes(span, opts.attrs);
        setAttributes(span, opts.attrsFn?.(...args));

        const onSuccess = (result: TResult): TResult => {
          setAttributes(span, opts.resultAttrsFn?.(result));
          if (opts.setOkStatus) span.setStatus({ code: SpanStatusCode.OK });
          return result;
        };

        const onError = (err: unknown): never => {
          handleSpanError(span, err, opts.isExpectedError);
          throw err;
        };

        try {
          const result = original.apply(this, args);
          if (isPromiseLike<TResult>(result)) {
            isAsync = true;
            return result
              .then(onSuccess)
              .catch((err) => onError(err))
              .finally(() => {
                span.end();
              });
          }

          return onSuccess(result);
        } catch (err) {
          return onError(err);
        } finally {
          if (!isAsync) {
            span.end();
          }
        }
      });
    };

    descriptor.value = wrapped as (...args: TArgs) => TResult | Promise<TResult>;
  } as MethodDecorator;
}
