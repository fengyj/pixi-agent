import { nanoid } from 'nanoid';
import { z } from 'zod';

const ModelCapabilitiesSchema = z.union([
  z.literal('reasoning'),
  z.literal('vision'),
  z.literal('image_generation'),
  z.literal('audio_input'),
  z.literal('audio_output'),
  z.literal('video_input'),
  z.literal('video_generation'),
  z.literal('function_calling'),
  z.literal('streaming'),
  z.literal('json_mode'),
  z.literal('web_search'),
  z.literal('code_execution'),
  z.literal('computer_use'),
]);

export type ModelCapabilities = z.infer<typeof ModelCapabilitiesSchema>;

export const ModelProviderSchema = z.union([
    z.literal('openai'),
    z.literal('anthropic'),
    z.literal('google'),
    z.literal('deepseek'),
    z.literal('alibaba'),
    z.literal('azure'),
    z.literal('openrouter'),
    z.literal('ofoxai'),
    z.string(),
  ]);

export type ModelProvider = z.infer<typeof ModelProviderSchema>;

export const ModelInfoSchema = z.object({
  modelId: z.string(),
  model: z.string(),
  displayName: z.string().optional(),
  description: z.string().optional(),
  baseUrl: z.string().optional(),
  apiKey: z.string().optional(),
  apiProtocol: z
    .union([
      z.literal('completions'),
      z.literal('response'),
      z.literal('anthropic'),
      z.literal('bedrock'),
    ])
    .optional(),
  provider: ModelProviderSchema,
  capabilities: z.array(ModelCapabilitiesSchema).optional(),
  contextWindow: z.number(),
  maxOutputTokens: z.number().optional(),
  pricing: z.object({
    free: z.boolean().optional(),
    currency: z.union([z.literal('USD'), z.literal('CNY'), z.string()]).optional(),
    standard: z.object({
      inputPerMillion: z.number(),
      outputPerMillion: z.number(),
      cacheReadPerMillion: z.number().optional(),
      cacheWritePerMillion: z.number().optional(),
    }),
  }),
});

export type ModelInfo = z.infer<typeof ModelInfoSchema>;

export class ModelRegistry {
  private models: Map<string, ModelInfo> = new Map();

  registerModel(modelInfo: Omit<ModelInfo, 'modelId'>): void {
    const modelId = nanoid(8);
    this.models.set(modelId, { ...modelInfo, modelId });
  }

  getModel(modelId: string): ModelInfo | undefined {
    return this.models.get(modelId);
  }

  listModels(): ModelInfo[] {
    return Array.from(this.models.values());
  }

  getModels(modelName: string, baseUrl?: string): ModelInfo[] {
    const matchedModels = Array.from(this.models.values()).filter(
      (model) => model.model === modelName && (baseUrl ? model.baseUrl === baseUrl : true),
    );
    return matchedModels;
  }
}
