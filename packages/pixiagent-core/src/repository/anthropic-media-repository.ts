import { Anthropic } from '@anthropic-ai/sdk';
import type { LLMProviderMediaRepository, LLMProviderUploadResponse } from './media-repository';

export class AnthropicMediaRepository implements LLMProviderMediaRepository {
  public readonly provider = 'anthropic';
  private readonly client: Anthropic;

  constructor(apiKey: string) {
    this.client = new Anthropic({ apiKey });
  }

  isSupported(apiBaseUrl: string, _model: string): boolean {
    const normalizedBaseUrl = apiBaseUrl?.toLowerCase() ?? '';
    return normalizedBaseUrl.startsWith('https://api.anthropic.com');
  }

  async upload(
    data: Uint8Array,
    fileName?: string,
    mimeType?: string,
  ): Promise<LLMProviderUploadResponse> {
    const file = await Anthropic.toFile(data, fileName ?? 'file', {
      type: mimeType,
    });

    const result = await this.client.beta.files.upload({
      file,
    });

    return {
      fileId: result.id,
      provider: this.provider,
      expireAt: undefined,
      // Anthropic beta file upload currently does not expose an expiration field in the returned metadata.
      // `purpose` and `metadata` are accepted by the interface but not used by this implementation.
    };
  }

  async delete(fileId: string): Promise<void> {
    await this.client.beta.files.delete(fileId);
  }
}
