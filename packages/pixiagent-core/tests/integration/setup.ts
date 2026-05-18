import fs from 'node:fs';
import path from 'node:path';
import { afterAll } from 'vitest';
import { getIntegrationConfig } from './config';
import { Observation } from '../../src/observation';
const { setupObservability, shutdownObservability } = Observation;

function parseEnvFile(content: string): Record<string, string> {
  const entries: Record<string, string> = {};

  for (const rawLine of content.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }

    const separatorIndex = line.indexOf('=');
    if (separatorIndex <= 0) {
      continue;
    }

    const key = line.slice(0, separatorIndex).trim();
    let value = line.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    entries[key] = value;
  }

  return entries;
}

function loadIntegrationEnv(): void {
  const envFile = path.resolve(process.cwd(), '.env.integration');
  if (!fs.existsSync(envFile)) {
    return;
  }

  const parsed = parseEnvFile(fs.readFileSync(envFile, 'utf8'));
  for (const [key, value] of Object.entries(parsed)) {
    if (process.env[key] === undefined) {
      process.env[key] = value;
    }
  }
}

loadIntegrationEnv();

// Initialise the observation module (OTel SDK + pino → collector) when
// OTEL_ENABLED=true is set in .env.integration.  Other tests automatically
// get structured JSON logs and no-op traces/metrics if this is disabled.
const _cfg = getIntegrationConfig();
if (_cfg.otelEnabled) {
  void setupObservability({
    transport: _cfg.otelTransport,
    transportEndpoint: _cfg.otelEndpoint,
    logging: {
      serviceName: _cfg.otelServiceName,
      serviceVersion: '0.1.0',
      outputToOtel: true,
      outputToConsole: true,
    },
  });

  afterAll(async () => {
    await shutdownObservability();
  });
}
