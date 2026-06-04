import type { ModelProvider } from '../model';

export interface WebUpdateResponse {
  url: string;
}

export interface WebPresignedUrlResponse {
  presignedUrl?: string;
  expiresAt?: number;
}

/**
 * The repository for the media accessed from a permanent URL or a presigned URL.
 */
export interface WebMediaRepository {
  /**
   * Uploads a media file to the repository.
   *
   * The URL of the file can use any of the parameters to determine, or none of them.
   * @param mediaId The unique identifier for the media.
   * @param data The media file data as a Uint8Array.
   * @param fileName Optional file name for the media.
   * @param path Optional path where the media should be stored.
   */
  upload(
    mediaId: string,
    data: Uint8Array,
    fileName?: string,
    path?: string,
  ): Promise<WebUpdateResponse>;

  /**
   * Downloads a media file from the repository.
   *
   * The repo can return the media data based on the mediaId, or the url.
   * Depends on the implementation.
   * @param mediaId The unique identifier for the media.
   * @param url The URL from which to download the media.
   */
  download(mediaId: string, url: string): Promise<Uint8Array>;

  /**
   * If the url is not accessible by LLM provider, this function will return a presigned url.
   * Otherwise, return a empty object `{}`.
   * @param mediaId
   * @param url
   */
  getPresignedUrl(mediaId: string, url: string): Promise<WebPresignedUrlResponse>;

  /** Delete the media from the repository */
  delete(mediaId: string, url: string): Promise<void>;
}

export interface LLMProviderUploadResponse {
  fileId: string;
  provider: ModelProvider;
  expireAt?: number;
}

/**
 * The wrapper for the LLM provider's file service.
 */
export interface LLMProviderMediaRepository {
  provider: ModelProvider;
  /**
   * Checks if the LLM provider supports the given model and baseUrl for media upload.
   * @param apiBaseUrl
   * @param model
   */
  isSupported(apiBaseUrl: string, model: string): boolean;

  upload(
    data: Uint8Array,
    fileName?: string,
    mimeType?: string,
  ): Promise<LLMProviderUploadResponse>;

  delete(fileId: string): Promise<void>;
}
