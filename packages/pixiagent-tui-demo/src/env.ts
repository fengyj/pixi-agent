import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ApiModes, Transports, Observation } from '@pixiagent/core';

type ObservabilityOptions = Observation.ObservabilityOptions;

type DemoObservabilityConfig = {
  enabled: boolean;
  options: ObservabilityOptions;
};

export type DemoConfig = {
  modelOptions: Transports.ModelOptions;
  modelRequestTimeout: number;
  maxIterations: number;
  maxModelRequestRetries: number;
  apiKeyVarName: string;
  observability: DemoObservabilityConfig;
};

function parseDotEnv(content: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const index = line.indexOf('=');
    if (index <= 0) continue;

    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    env[key] = value;
  }
  return env;
}

function normalizeApiMode(value: string | undefined): ApiModes | undefined {
  if (!value) return undefined;
  const normalized = value.toLowerCase();
  if (normalized === 'completions') return ApiModes.COMPLETIONS;
  if (normalized === 'response') return ApiModes.RESPONSE;
  if (normalized === 'anthropic') return ApiModes.ANTHROPIC;
  return undefined;
}

function parseNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
  return Math.floor(parsed);
}

function parseBoolean(value: string | undefined, fallback: boolean): boolean {
  if (!value) return fallback;
  const normalized = value.trim().toLowerCase();
  if (['1', 'true', 'yes', 'y', 'on'].includes(normalized)) return true;
  if (['0', 'false', 'no', 'n', 'off'].includes(normalized)) return false;
  return fallback;
}

function normalizeTransport(value: string | undefined): ObservabilityOptions['transport'] {
  const normalized = value?.trim().toLowerCase();
  if (normalized === 'grpc' || normalized === 'http' || normalized === 'none') {
    return normalized;
  }
  return 'none';
}

function getEnvFilePath(): string {
  const fileDir = fileURLToPath(new URL('.', import.meta.url));
  const packageRoot = resolve(fileDir, '..');
  const localEnv = resolve(packageRoot, '.env');
  if (existsSync(localEnv)) return localEnv;
  return resolve(process.cwd(), '.env');
}

export function loadConfigFromEnv(): DemoConfig {
  const envPath = getEnvFilePath();
  const localEnv = existsSync(envPath) ? parseDotEnv(readFileSync(envPath, 'utf8')) : {};

  const merged = {
    ...localEnv,
    ...process.env,
  } as Record<string, string | undefined>;

  const model = merged.PIXIA_MODEL;
  const baseUrl = merged.PIXIA_BASE_URL;
  const apiKeyVarName = merged.PIXIA_API_KEY_ENV;
  const systemPrompt = merged.PIXIA_SYSTEM_PROMPT;

  if (!model) {
    throw new Error('Missing PIXIA_MODEL in .env');
  }
  if (!baseUrl) {
    throw new Error('Missing PIXIA_BASE_URL in .env');
  }
  if (!apiKeyVarName) {
    throw new Error('Missing PIXIA_API_KEY_ENV in .env');
  }

  const apiKey = merged[apiKeyVarName];
  if (!apiKey) {
    throw new Error(`Missing API key in env variable: ${apiKeyVarName}`);
  }

  const apiMode = normalizeApiMode(merged.PIXIA_API_MODE);
  const otelEnabled = parseBoolean(merged.PIXIA_OTEL_ENABLED, false);
  const otelTransport = normalizeTransport(merged.PIXIA_OTEL_TRANSPORT);
  const otelEndpoint = merged.PIXIA_OTEL_ENDPOINT;
  const otelEnableTelemetry = parseBoolean(merged.PIXIA_OTEL_ENABLE_TELEMETRY, otelTransport !== 'none');
  const otelOutputToOtel = parseBoolean(merged.PIXIA_OTEL_OUTPUT_TO_OTEL, otelTransport !== 'none');

  if (otelEnabled && otelTransport !== 'none' && !otelEndpoint) {
    throw new Error('Missing PIXIA_OTEL_ENDPOINT in .env when PIXIA_OTEL_TRANSPORT is grpc/http');
  }

  const normalizedOtelEndpoint =
    otelTransport === 'http' && otelEndpoint && !/^https?:\/\//i.test(otelEndpoint)
      ? `http://${otelEndpoint}`
      : otelEndpoint;

  return {
    modelOptions: {
      model,
      baseUrl,
      apiKey,
      apiMode,
      ...(systemPrompt ? { systemPrompt } : {}),
    },
    modelRequestTimeout: parseNumber(merged.PIXIA_MODEL_TIMEOUT_MS, 120_000),
    maxIterations: parseNumber(merged.PIXIA_MAX_ITERATIONS, 8),
    maxModelRequestRetries: parseNumber(merged.PIXIA_MAX_MODEL_RETRIES, 1),
    apiKeyVarName,
    observability: {
      enabled: otelEnabled,
      options: {
        transport: otelTransport,
        transportEndpoint: normalizedOtelEndpoint,
        enableTelemetry: otelEnableTelemetry,
        logging: {
          serviceName: merged.PIXIA_OTEL_SERVICE_NAME ?? 'pixiagent-tui-demo',
          serviceVersion: merged.PIXIA_OTEL_SERVICE_VERSION ?? '0.1.0',
          outputToConsole: false,
          outputToOtel: otelOutputToOtel,
        },
      },
    },
  };
}
