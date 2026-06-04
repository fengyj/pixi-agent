import { OpenAI } from 'openai/client';
import type { LLMProviderMediaRepository, LLMProviderUploadResponse } from './media-repository';

export class OpenAIMediaRepository implements LLMProviderMediaRepository {
  public readonly provider = 'openai';
  private readonly client: OpenAI;

  constructor(apiKey: string) {
    this.client = new OpenAI({ apiKey });
  }

  isSupported(apiBaseUrl: string, _model: string): boolean {
    const normalizedBaseUrl = apiBaseUrl?.toLowerCase() ?? '';
    return normalizedBaseUrl.startsWith('https://api.openai.com');
  }

  async upload(
    data: Uint8Array,
    fileName?: string,
    mimeType?: string,
  ): Promise<LLMProviderUploadResponse> {
    const file = await OpenAI.toFile(data, fileName ?? 'file', {
      type: mimeType,
    });

    const result = await this.client.files.create({
      file,
      purpose: 'user_data' as OpenAI.FilePurpose,
    });

    return {
      fileId: result.id,
      provider: this.provider,
      expireAt: result.expires_at,
      // OpenAI file upload does not support arbitrary metadata in the file create endpoint,
      // so metadata is accepted by the interface but not persisted here.
    };
  }

  async delete(fileId: string): Promise<void> {
    await this.client.files.delete(fileId);
  }
}
