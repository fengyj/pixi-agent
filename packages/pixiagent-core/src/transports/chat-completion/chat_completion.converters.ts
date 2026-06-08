import type {
  ChatCompletionUserMessageParam,
  ChatCompletionAssistantMessageParam,
  ChatCompletionToolMessageParam,
  ChatCompletionMessageParam,
  ChatCompletionContentPartText,
  ChatCompletionContentPartRefusal,
  ChatCompletionContentPartImage,
  ChatCompletionContentPartInputAudio,
  ChatCompletionContentPart,
  ChatCompletionFunctionMessageParam,
  ChatCompletionMessage,
} from 'openai/resources/chat/completions';
import {
  ApiModes,
  ChatCompletionApiMessage,
  SessionMessage,
  ToolCallPart,
  TextPart,
  RefusalPart,
  AudioPart,
  ImagePart,
  DocumentPart,
  ToolResultPart,
  ContentPart,
  CitationWebLocation,
} from '../../message';
import { PixiAgentErrorBuilder } from '../../errors';
import { assertNever } from '../../utils';

export class ChatCompletionMessageConverter {
  private readonly rawConverter = new ChatCompletionRawMessageConverter();
  private readonly sessionConverter = new ChatCompletionSessionMessageConverter();

  convertFromRawMessage(rawMsg: ChatCompletionApiMessage): SessionMessage {
    return this.rawConverter.convertFromRawMessage(rawMsg);
  }

  convertToRawMessage(msg: SessionMessage): ChatCompletionApiMessage | ChatCompletionApiMessage[] {
    return this.sessionConverter.convertToRawMessage(msg);
  }
}

class ChatCompletionRawMessageConverter {
  convertFromRawMessage(rawMsg: ChatCompletionApiMessage): SessionMessage {
    switch (rawMsg.content.role) {
      case 'assistant':
        return this.getFromAssistantMessageParam(rawMsg);
      case 'user':
        return this.getFromUserMessageParam(rawMsg);
      case 'tool':
        return this.getFromToolMessageParam(rawMsg);
      case 'function':
        return this.getFromFunctionMessageParam(rawMsg);
      case 'developer':
      case 'system':
        throw PixiAgentErrorBuilder.invalidMessage(
          `Unsupported message role: ${rawMsg.content.role}`,
        );
      default:
        assertNever(rawMsg.content);
    }
  }

  private getFromAssistantMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionAssistantMessageParam;

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'assistant',
      content: this.buildAssistantParts(msg),
      name: msg.name,
      modelResponseInfo: rawMsg.modelResponseInfo,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private buildAssistantParts(msg: ChatCompletionAssistantMessageParam): ContentPart[] | string {
    const parts: ContentPart[] = [];

    this.appendFunctionCallParts(parts, msg.function_call, msg.tool_calls);

    if (Array.isArray(msg.content)) {
      msg.content.forEach((contentPart) => {
        if (contentPart.type === 'text') {
          parts.push(this.convertFromChatCompletionTextPart(contentPart));
        } else if (contentPart.type === 'refusal') {
          parts.push(this.convertFromChatCompletionRefusalPart(contentPart));
        }
      });
    } else {
      const annotations =
        'annotations' in msg && Array.isArray(msg.annotations)
          ? this.convertAnnotations(msg.annotations)
          : undefined;

      if (typeof msg.content === 'string' && msg.content.length > 0) {
        if (
          !('refusal' in msg && msg.refusal) &&
          parts.length === 0 &&
          (annotations === undefined || annotations.length === 0)
        ) {
          return msg.content;
        } else {
          parts.push({ type: 'text', text: msg.content, citations: annotations } as TextPart);
        }
      }
      if ('refusal' in msg && msg.refusal) {
        parts.push({ type: 'refusal', reason: msg.refusal } as RefusalPart);
      }
    }

    return parts;
  }

  private appendFunctionCallParts(
    parts: ContentPart[],
    legacyFunctionCall: ChatCompletionAssistantMessageParam['function_call'],
    toolCalls: ChatCompletionAssistantMessageParam['tool_calls'],
  ): void {
    if (legacyFunctionCall) {
      parts.push({
        type: 'tool_call',
        id: '',
        name: legacyFunctionCall.name,
        arguments: legacyFunctionCall.arguments,
      } as ToolCallPart);
    }

    // Some providers return non-standard tool call payloads in `tool_calls`.
    // Keep those as provider-specific tool_call parts so the rest of the stack can preserve them.
    toolCalls?.forEach((toolCall) => {
      if (toolCall.type === 'function') {
        parts.push({
          type: 'tool_call',
          id: toolCall.id,
          name: toolCall.function.name,
          arguments: toolCall.function.arguments,
        } as ToolCallPart);
        return;
      }

      const { id, type, ...rest } = toolCall;
      parts.push({
        type: 'tool_call',
        id,
        name: type,
        arguments: JSON.stringify(rest),
        providerSpecific: ApiModes.COMPLETIONS,
      } as ToolCallPart);
    });
  }

  private toSimpleAssistantSessionMessage(
    rawMsg: ChatCompletionApiMessage,
    msg: ChatCompletionAssistantMessageParam,
  ): SessionMessage {
    const annotations =
      'annotations' in msg && Array.isArray(msg.annotations)
        ? this.convertAnnotations(msg.annotations)
        : undefined;
    if ('refusal' in msg && msg.refusal) {
      const contents = [];
      contents.push({ type: 'refusal', reason: msg.refusal } as RefusalPart);
      if (typeof msg.content === 'string' && msg.content.length > 0) {
        contents.push({ type: 'text', text: msg.content, citations: annotations } as TextPart);
      }
      return {
        messageId: rawMsg.messageId,
        type: 'session_message',
        role: 'assistant',
        content: contents,
        name: msg.name,
        metadata: rawMsg.metadata,
      } as SessionMessage;
    }
    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'assistant',
      content: msg.content,
      name: msg.name,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromUserMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionUserMessageParam;
    if (typeof msg.content === 'string') {
      return {
        messageId: rawMsg.messageId,
        type: 'session_message',
        role: 'user',
        content: msg.content,
        name: msg.name,
        metadata: rawMsg.metadata,
      } as SessionMessage;
    }

    const parts: ContentPart[] = [];
    msg.content.forEach((contentPart) => {
      if (contentPart.type === 'text') {
        parts.push(this.convertFromChatCompletionTextPart(contentPart));
      } else if (contentPart.type === 'image_url') {
        parts.push(this.convertFromChatCompletionImagePart(contentPart));
      } else if (contentPart.type === 'input_audio') {
        parts.push(this.convertFromChatCompletionAudioPart(contentPart));
      } else if (contentPart.type === 'file') {
        const filePart = this.convertFromChatCompletionFilePart(contentPart);
        if (filePart) {
          parts.push(filePart);
        }
      }
    });

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'user',
      content: parts,
      name: msg.name,
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromToolMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionToolMessageParam;
    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw PixiAgentErrorBuilder.invalidMessage('Tool message must have content', 'tool');
    }

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'tool',
      content: [
        {
          type: 'tool_result',
          id: msg.tool_call_id,
          result: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
        },
      ],
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private getFromFunctionMessageParam(rawMsg: ChatCompletionApiMessage): SessionMessage {
    const msg = rawMsg.content as ChatCompletionFunctionMessageParam;
    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: 'assistant',
      content: [
        {
          type: 'tool_call',
          id: '',
          name: msg.name,
          arguments: msg.content,
        } as ToolCallPart,
      ],
      metadata: rawMsg.metadata,
    } as SessionMessage;
  }

  private convertFromChatCompletionTextPart(part: ChatCompletionContentPartText): TextPart {
    return { type: 'text', text: part.text };
  }

  private convertAnnotations(
    annotations?: Array<ChatCompletionMessage.Annotation>,
  ): Array<CitationWebLocation> {
    if (!annotations || annotations.length === 0) {
      return [];
    }

    return annotations
      .filter(
        (annotation): annotation is ChatCompletionMessage.Annotation =>
          annotation.type === 'url_citation',
      )
      .map((annotation) => ({
        type: 'web_location' as const,
        url: annotation.url_citation.url,
        citedText: '',
        title: annotation.url_citation.title,
        startIndex: annotation.url_citation.start_index,
        endIndex: annotation.url_citation.end_index,
        extra: { rawCitationType: 'url_citation' },
      }));
  }

  private attachCitationsToTextParts(
    parts: ContentPart[],
    citations: Array<CitationWebLocation>,
  ): void {
    if (citations.length === 0) {
      return;
    }

    const firstText = parts.find((part) => part.type === 'text') as TextPart | undefined;
    if (firstText) {
      firstText.citations = [...(firstText.citations ?? []), ...citations];
      return;
    }

    parts.unshift({
      type: 'text',
      text: '',
      citations,
    } as TextPart);
  }

  private convertFromChatCompletionRefusalPart(
    part: ChatCompletionContentPartRefusal,
  ): RefusalPart {
    return { type: 'refusal', reason: part.refusal };
  }

  private convertFromChatCompletionImagePart(part: ChatCompletionContentPartImage): ImagePart {
    const isBase64 = part.image_url.url.startsWith('data:image/');
    if (isBase64) {
      const mediaType = part.image_url.url.split(';')[0].split(':')[1];
      const base64Data = part.image_url.url.split(',')[1];
      return {
        type: 'image',
        image: {
          sourceType: 'base64',
          data: base64Data,
          mimeType: mediaType,
        },
      };
    }

    return {
      type: 'image',
      image: {
        sourceType: 'url',
        url: part.image_url.url,
      },
    };
  }

  private convertFromChatCompletionAudioPart(part: ChatCompletionContentPartInputAudio): AudioPart {
    return {
      type: 'audio',
      audio: {
        sourceType: 'base64',
        data: part.input_audio.data,
        mimeType: `audio/${part.input_audio.format}`,
      },
    };
  }

  private convertFromChatCompletionFilePart(
    part: ChatCompletionContentPart.File,
  ): DocumentPart | null {
    if (part.file.file_data) {
      return {
        type: 'document',
        document: {
          sourceType: 'base64',
          mimeType: 'application/octet-stream',
          data: part.file.file_data,
          fileName: part.file.filename ?? undefined,
        },
      };
    }

    if (part.file.file_id) {
      return {
        type: 'document',
        document: {
          sourceType: 'file_id',
          fileId: part.file.file_id,
          fileName: part.file.filename ?? undefined,
        },
      };
    }

    return null;
  }
}

class ChatCompletionSessionMessageConverter {
  convertToRawMessage(msg: SessionMessage): ChatCompletionApiMessage | ChatCompletionApiMessage[] {
    const raw = (() => {
      switch (msg.role) {
        case 'assistant':
          return this.getAssistantMessageParam(msg);
        case 'user':
          return this.getUserMessageParam(msg);
        case 'tool':
          return this.getToolMessageParam(msg);
        default:
          assertNever(msg as never);
      }
    })();

    const wrap = (inner: ChatCompletionMessageParam): ChatCompletionApiMessage => ({
      messageId: msg.messageId,
      type: 'chat_completion_api_message',
      role:
        inner.role === 'function'
          ? 'tool'
          : inner.role === 'developer' || inner.role === 'system'
            ? 'user'
            : inner.role,
      content: inner,
      modelResponseInfo: msg.modelResponseInfo,
      metadata: msg.metadata,
    });

    if (Array.isArray(raw)) {
      const wrapped = raw.map(wrap);
      return wrapped.length === 1 ? wrapped[0] : wrapped;
    }

    return wrap(raw);
  }

  private getAssistantMessageParam(msg: SessionMessage): ChatCompletionAssistantMessageParam {
    if (msg.role !== 'assistant') {
      throw PixiAgentErrorBuilder.invalidMessage(
        `Message role must be assistant, but got ${msg.role}`,
      );
    }

    const toolCalls = this.toFunctionToolCalls(msg);
    const { content, refusal } = this.toAssistantContent(msg);

    return {
      role: 'assistant',
      content: content,
      refusal: refusal,
      tool_calls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
      audio: null,
      name: msg.name,
    } as ChatCompletionAssistantMessageParam;
  }

  private toFunctionToolCalls(msg: SessionMessage) {
    return msg.content instanceof Array
      ? msg.content
          .filter((part) => part.type === 'tool_call')
          .map((part) => part as ToolCallPart)
          .map((part) => {
            try {
              // If the caller explicitly marked the call as provider-specific,
              // preserve the original object shape instead of normalizing it.
              if (part.providerSpecific === ApiModes.COMPLETIONS) {
                const parsedArguments = JSON.parse(part.arguments);
                if (parsedArguments && typeof parsedArguments === 'object') {
                  return {
                    ...parsedArguments,
                    type: part.name,
                    id: part.id,
                  };
                }
              }
            } catch {
              // Keep the default function-call conversion path below.
            }

            return {
              type: 'function',
              id: part.id,
              function: {
                name: part.name,
                arguments: part.arguments,
              },
            };
          })
      : undefined;
  }

  private toAssistantContent(msg: SessionMessage): {
    content:
      | Array<ChatCompletionContentPartText | ChatCompletionContentPartRefusal>
      | string
      | null;
    refusal?: string;
  } {
    if (!(msg.content instanceof Array)) {
      return { content: msg.content ?? null, refusal: undefined };
    }

    const parts = msg.content
      .map((part) => this.convertToChatCompletionAssistantContentPart(part))
      .filter(
        (part): part is ChatCompletionContentPartText | ChatCompletionContentPartRefusal =>
          part !== null,
      );
    const textParts = parts.filter(
      (part) => part.type === 'text',
    ) as ChatCompletionContentPartText[];
    const refusalParts = parts.filter(
      (part) => part.type === 'refusal',
    ) as ChatCompletionContentPartRefusal[];
    return {
      content:
        textParts.length <= 1 && refusalParts.length <= 1 ? (textParts[0]?.text ?? null) : parts,
      refusal: refusalParts[0]?.refusal,
    };
  }

  private getUserMessageParam(msg: SessionMessage): ChatCompletionUserMessageParam {
    if (msg.role !== 'user') {
      throw PixiAgentErrorBuilder.invalidMessage(`Message role must be user, but got ${msg.role}`);
    }

    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw PixiAgentErrorBuilder.invalidMessage('User message must have content', 'user');
    }

    if (typeof msg.content === 'string') {
      return {
        role: 'user',
        content: msg.content,
        name: msg.name,
      } as ChatCompletionUserMessageParam;
    }

    const content = msg.content
      .map((part) => this.convertToChatCompletionUserContentPart(part))
      .filter((part): part is ChatCompletionContentPart => part !== null);

    return {
      role: 'user',
      content: content.length === 0 ? null : content,
      name: msg.name,
    } as ChatCompletionUserMessageParam;
  }

  private getToolMessageParam(
    msg: SessionMessage,
  ): ChatCompletionMessageParam | ChatCompletionMessageParam[] {
    if (msg.role !== 'tool') {
      throw PixiAgentErrorBuilder.invalidMessage(`Message role must be tool, but got ${msg.role}`);
    }

    if (!msg.content || typeof msg.content === 'string' || msg.content.length === 0) {
      throw PixiAgentErrorBuilder.invalidMessage(
        'Tool message must have content that is a non-empty array.',
        'tool',
      );
    }

    const toolResults = msg.content.filter(
      (part) => part.type === 'tool_result',
    ) as ToolResultPart[];
    const otherParts = msg.content.filter((part) => part.type !== 'tool_result');

    if (toolResults.length === 0) {
      throw PixiAgentErrorBuilder.invalidMessage(
        'Tool message content must have at least one tool result part.',
        'tool',
      );
    }

    const rawMessages: ChatCompletionMessageParam[] = toolResults.map((toolResult) => {
      try {
        const resultObj =
          !toolResult.result || toolResult.result === ''
            ? null
            : JSON.parse(toolResult.result ?? '{}');

        if (
          Array.isArray(resultObj) &&
          resultObj.every(
            (item) =>
              typeof item === 'object' &&
              item !== null &&
              'type' in item &&
              item.type === 'text' &&
              'text' in item &&
              typeof item.text === 'string',
          )
        ) {
          return {
            role: 'tool',
            tool_call_id: toolResult.id,
            content: resultObj.map((item: { text: string }) => ({ type: 'text', text: item.text })),
          };
        }

        return {
          role: 'tool',
          tool_call_id: toolResult.id,
          content: toolResult.result ?? JSON.stringify(null),
        };
      } catch {
        return {
          role: 'tool',
          tool_call_id: toolResult.id,
          content: toolResult.result ?? JSON.stringify(null),
        };
      }
    });

    if (otherParts.length > 0) {
      rawMessages.push(
        this.getUserMessageParam({
          ...msg,
          role: 'user',
          content: otherParts,
        } as SessionMessage),
      );
    }

    return rawMessages.length === 1 ? rawMessages[0] : rawMessages;
  }

  private convertToChatCompletionTextPart(part: TextPart): ChatCompletionContentPartText {
    return { type: 'text', text: part.text };
  }

  private convertToChatCompletionRefusalPart(part: RefusalPart): ChatCompletionContentPartRefusal {
    return { type: 'refusal', refusal: part.reason };
  }

  private convertToChatCompletionImagePart(part: ImagePart): ChatCompletionContentPartImage | null {
    switch (part.image.sourceType) {
      case 'url':
        return { type: 'image_url', image_url: { url: part.image.url } };
      case 'base64':
        return {
          type: 'image_url',
          image_url: { url: `data:${part.image.mimeType};base64,${part.image.data}` },
        };
      default:
        return null;
    }
  }

  private convertToChatCompletionAudioPart(
    part: AudioPart,
  ): ChatCompletionContentPartInputAudio | null {
    if (part.audio.sourceType !== 'base64') {
      return null;
    }

    const format = part.audio.mimeType.replace('audio/', '');
    return {
      type: 'input_audio',
      input_audio: {
        data: part.audio.data,
        format: format as 'wav' | 'mp3',
      },
    };
  }

  private convertToChatCompletionFilePart(part: DocumentPart): ChatCompletionContentPart.File {
    return {
      type: 'file',
      file: {
        file_data: part.document.sourceType === 'base64' ? part.document.data : undefined,
        file_id: part.document.sourceType === 'file_id' ? part.document.fileId : undefined,
        filename: part.document.fileName ?? undefined,
      },
    };
  }

  private convertToChatCompletionAssistantContentPart(
    part: ContentPart,
  ): ChatCompletionContentPartText | ChatCompletionContentPartRefusal | null {
    if (part.type === 'text') {
      return this.convertToChatCompletionTextPart(part as TextPart);
    }
    if (part.type === 'refusal') {
      return this.convertToChatCompletionRefusalPart(part as RefusalPart);
    }
    return null;
  }

  private convertToChatCompletionUserContentPart(
    part: ContentPart,
  ): ChatCompletionContentPart | null {
    switch (part.type) {
      case 'text':
        return this.convertToChatCompletionTextPart(part as TextPart);
      case 'image':
        return this.convertToChatCompletionImagePart(part as ImagePart);
      case 'audio':
        return this.convertToChatCompletionAudioPart(part as AudioPart);
      case 'document':
        return this.convertToChatCompletionFilePart(part as DocumentPart);
      default:
        return null;
    }
  }
}
