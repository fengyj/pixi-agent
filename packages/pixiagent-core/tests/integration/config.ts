import { z } from 'zod';

const IntegrationConfigSchema = z.object({
  openaiApiKey: z.string().min(1).optional(),
  anthropicApiKey: z.string().min(1).optional(),
  baseUrl: z.string().url().optional(),
  model: z.string().min(1).default('gpt-4.1-mini'),
  runLiveTests: z.coerce.boolean().default(false),
  // OTel / observation
  otelEnabled: z.coerce.boolean().default(false),
  otelTransport: z.enum(['grpc', 'http', 'none']).default('grpc'),
  otelEndpoint: z.string().optional(),
  otelServiceName: z.string().default('pixiagent-integration-test'),
});

export type IntegrationConfig = z.infer<typeof IntegrationConfigSchema>;

export function getIntegrationConfig(): IntegrationConfig {
  return IntegrationConfigSchema.parse({
    openaiApiKey: process.env.OPENAI_API_KEY,
    anthropicApiKey: process.env.ANTHROPIC_API_KEY,
    baseUrl: process.env.LLM_BASE_URL,
    model: process.env.LLM_MODEL,
    runLiveTests: process.env.RUN_LIVE_INTEGRATION_TESTS,
    otelEnabled: process.env.OTEL_ENABLED,
    otelTransport: process.env.OTEL_TRANSPORT,
    otelEndpoint: process.env.OTEL_ENDPOINT,
    otelServiceName: process.env.OTEL_SERVICE_NAME,
  });
}