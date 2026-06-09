import type {
  TextBlockParam,
  TextCitation,
  TextCitationParam,
  ImageBlockParam,
  DocumentBlockParam,
  ToolUseBlockParam,
  ToolResultBlockParam,
  ThinkingBlockParam,
  ServerToolUseBlockParam,
  ToolReferenceBlockParam,
  SearchResultBlockParam,
  ContentBlock,
  ContentBlockParam,
  ToolReferenceBlock,
  ToolSearchToolResultBlock,
  ToolSearchToolResultBlockParam,
  WebFetchToolResultBlock,
  WebFetchToolResultBlockParam,
  WebSearchToolResultBlock,
  WebSearchToolResultBlockParam,
  TextEditorCodeExecutionToolResultBlock,
  TextEditorCodeExecutionToolResultBlockParam,
  CodeExecutionToolResultBlock,
  CodeExecutionToolResultBlockParam,
  BashCodeExecutionToolResultBlock,
  BashCodeExecutionToolResultBlockParam,
  ContainerUploadBlock,
  ContainerUploadBlockParam,
  ServerToolUseBlock,
  TextBlock,
  ThinkingBlock,
  ToolUseBlock,
} from '@anthropic-ai/sdk/resources/messages/messages';
import {
  AnthropicApiMessage,
  ApiModes,
  SessionMessage,
  ContentPart,
  TextPart,
  ThinkingPart,
  ToolCallPart,
  ToolResultPart,
  ImagePart,
  DocumentPart,
  RefusalPart,
  Citation,
  ServerToolUsePart,
  AudioPart,
  VideoPart,
} from '../../message';
import { assertNever } from '../../utils';

class AnthropicRawMessageConverter {
  convertFromRawMessage(rawMsg: AnthropicApiMessage): SessionMessage {
    const inner = rawMsg.content;

    return {
      messageId: rawMsg.messageId,
      type: 'session_message',
      role: rawMsg.role,
      content:
        typeof inner.content === 'string'
          ? inner.content
          : inner.content.map((block) => blockPartConverter.toParts(block)).flat(),
      modelResponseInfo: rawMsg.modelResponseInfo,
      metadata: rawMsg.metadata,
    };
  }
}

class AnthropicSessionMessageConverter {
  convertToRawMessage(msg: SessionMessage): AnthropicApiMessage {
    const content = {
      role: msg.role === 'tool' ? 'user' : msg.role,
      content:
        typeof msg.content === 'string'
          ? msg.content
          : msg.content
              .map((part) => partConverter.toBlockParam(part))
              .filter((part): part is ContentBlockParam => part !== null),
    };

    return {
      messageId: msg.messageId,
      type: 'anthropic_api_message',
      role: msg.role,
      content,
      modelResponseInfo: msg.modelResponseInfo,
      metadata: msg.metadata,
    };
  }
}

class AnthropicBlockPartConverter {
  toParts(block: ContentBlockParam | ContentBlock | ToolReferenceBlockParam): ContentPart[] {
    const part = this.toSinglePart(block);
    return Array.isArray(part) ? part : [part];
  }

  private toSinglePart(
    block: ContentBlockParam | ContentBlock | ToolReferenceBlockParam,
  ): ContentPart | ContentPart[] {
    switch (block.type) {
      case 'text':
        return this.toTextPart(block);
      case 'thinking':
        return this.toThinkingPart(block);
      case 'image':
        return this.toImagePart(block);
      case 'document':
        return this.toDocumentPart(block);
      case 'tool_use':
      case 'server_tool_use':
        return this.toToolCallPart(block);
      case 'tool_result':
        return this.toToolResultPart(block);
      case 'container_upload':
        return this.toContainerDocumentPart(block);
      case 'bash_code_execution_tool_result':
      case 'code_execution_tool_result':
      case 'text_editor_code_execution_tool_result':
      case 'tool_search_tool_result':
      case 'web_fetch_tool_result':
      case 'web_search_tool_result':
        return this.toSpecificToolResultPart(block);
      case 'tool_reference':
        return this.toToolReferenceTextPart(block);
      case 'search_result':
        return this.toSearchResultTextPart(block);
      case 'redacted_thinking':
        return [];
      default:
        return assertNever(block as never);
    }
  }

  private toTextPart(block: TextBlockParam | TextBlock): TextPart {
    return {
      type: 'text',
      text: block.text,
      citations: block.citations?.map((citation) => this.toCitation(citation)),
    };
  }

  private toCitation(citation: TextCitation | TextCitationParam): Citation {
    switch (citation.type) {
      case 'char_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_char_index,
          endIndex: citation.end_char_index,
          indexType: 'char',
          source: `document index: ${citation.document_index}`,
        };
      case 'page_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_page_number,
          endIndex: citation.end_page_number,
          indexType: 'page',
          source: `document index: ${citation.document_index}`,
        };
      case 'content_block_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.document_title ?? undefined,
          startIndex: citation.start_block_index,
          endIndex: citation.end_block_index,
          indexType: 'block',
          source: `document index: ${citation.document_index}`,
        };
      case 'search_result_location':
        return {
          type: 'others_location',
          citedText: citation.cited_text,
          title: citation.title ?? undefined,
          startIndex: citation.start_block_index,
          endIndex: citation.end_block_index,
          indexType: 'block',
          source: citation.source,
        };
      case 'web_search_result_location':
        return {
          type: 'web_location',
          citedText: citation.cited_text,
          title: citation.title ?? undefined,
          url: citation.url,
        };
      default:
        return assertNever(citation as never);
    }
  }

  private toImagePart(block: ImageBlockParam): ImagePart {
    switch (block.source.type) {
      case 'base64':
        return {
          type: 'image',
          image: {
            sourceType: 'base64',
            mimeType: block.source.media_type,
            data: block.source.data,
          },
        };
      case 'url':
        return { type: 'image', image: { sourceType: 'url', url: block.source.url } };
      default:
        return assertNever(block.source as never);
    }
  }

  private toDocumentPart(block: DocumentBlockParam): DocumentPart | TextPart[] {
    switch (block.source.type) {
      case 'base64':
        return {
          type: 'document',
          document: {
            sourceType: 'base64',
            mimeType: block.source.media_type,
            data: block.source.data,
          },
        };
      case 'text':
        return [{ type: 'text', text: block.source.data }];
      case 'url':
        return { type: 'document', document: { sourceType: 'url', url: block.source.url } };
      case 'content':
        return (typeof block.source.content === 'string'
          ? [{ type: 'text', text: block.source.content }]
          : block.source.content
              .map((item) => this.toSinglePart(item as never))
              .flat()
              .filter((part): part is TextPart | ImagePart => part.type === 'text' || part.type === 'image')) as TextPart[];
      default:
        return assertNever(block.source as never);
    }
  }

  private toThinkingPart(block: ThinkingBlockParam | ThinkingBlock): ThinkingPart {
    return { type: 'thinking', content: block.thinking, signature: block.signature };
  }

  private toToolCallPart(
    block: ToolUseBlockParam | ToolUseBlock | ServerToolUseBlockParam | ServerToolUseBlock,
  ): ToolCallPart | ServerToolUsePart {
    switch (block.type) {
      case 'tool_use': {
        if (!block.caller || block.caller.type === 'direct') {
          return {
            type: 'tool_call',
            id: block.id,
            name: block.name,
            arguments: JSON.stringify(block.input ?? null),
          };
        }

        const { name, ...rest } = block as ToolUseBlock & { name: string };
        return {
          type: 'server_tool_use',
          name,
          data: JSON.stringify(rest),
          providerSpecific: ApiModes.ANTHROPIC,
        };
      }
      case 'server_tool_use': {
        if (block.caller && block.caller.type === 'direct') {
          const { id, name, ...rest } = block;
          return {
            type: 'tool_call',
            id,
            name,
            arguments: JSON.stringify(rest),
            providerSpecific: ApiModes.ANTHROPIC,
          };
        }

        const { name, type, ...restBlock } = block;
        return {
          type: type,
          name: name,
          data: JSON.stringify(restBlock),
          providerSpecific: ApiModes.ANTHROPIC,
        };
      }
      default:
        return assertNever(block as never);
    }
  }

  private toToolResultPart(block: ToolResultBlockParam): ToolResultPart {
    if (!block.content || typeof block.content === 'string') {
      return {
        type: 'tool_result',
        id: block.tool_use_id,
        result: typeof block.content === 'string' ? block.content : undefined,
        isError: block.is_error ?? undefined,
      };
    }

    return {
      type: 'tool_result',
      id: block.tool_use_id,
      result: JSON.stringify(
        block.content
          .map((item) => this.toSinglePart(item as never))
          .flat(),
      ),
      isError: block.is_error ?? undefined,
    };
  }

  private toContainerDocumentPart(block: ContainerUploadBlockParam | ContainerUploadBlock): DocumentPart {
    return {
      type: 'document',
      document: {
        sourceType: 'file_id',
        fileId: block.file_id,
      },
    };
  }

  private toSpecificToolResultPart(
    block:
      | BashCodeExecutionToolResultBlockParam
      | BashCodeExecutionToolResultBlock
      | CodeExecutionToolResultBlockParam
      | CodeExecutionToolResultBlock
      | TextEditorCodeExecutionToolResultBlockParam
      | TextEditorCodeExecutionToolResultBlock
      | ToolSearchToolResultBlockParam
      | ToolSearchToolResultBlock
      | WebFetchToolResultBlockParam
      | WebFetchToolResultBlock
      | WebSearchToolResultBlockParam
      | WebSearchToolResultBlock,
  ): ToolResultPart {
    const { tool_use_id, type, ...rest } = block;

    return {
      type: 'tool_result',
      id: tool_use_id,
      name: type.replace('_tool_result', ''),
      result: JSON.stringify(rest),
      providerSpecific: ApiModes.ANTHROPIC,
    };
  }

  private toToolReferenceTextPart(block: ToolReferenceBlockParam | ToolReferenceBlock): TextPart {
    return {
      type: 'text',
      text: JSON.stringify({ tool_name: block.tool_name, type: block.type }),
    };
  }

  private toSearchResultTextPart(block: SearchResultBlockParam): TextPart {
    return {
      type: 'text',
      text: JSON.stringify({
        source: block.source,
        title: block.title,
        content: block.content,
        type: block.type,
      }),
    };
  }
}

class AnthropicPartConverter {
  toBlockParam(part: ContentPart): ContentBlockParam | null {
    switch (part.type) {
      case 'text':
      case 'refusal':
        return this.toTextBlockParam(part);
      case 'thinking':
        return this.toThinkingBlockParam(part);
      case 'image':
        return this.toImageBlockParam(part);
      case 'document':
        return this.toDocumentBlockParam(part);
      case 'tool_call':
        return this.toToolUseBlockParam(part);
      case 'tool_result':
        return this.toToolResultBlockParam(part);
      case 'server_tool_use':
        return this.toServerToolUseBlockParam(part);
      case 'audio':
        return this.toAudioTextBlockParam(part);
      case 'video':
        return this.toVideoTextBlockParam(part);
      default:
        return assertNever(part as never);
    }
  }

  private toTextBlockParam(part: TextPart | RefusalPart): TextBlockParam {
    return {
      type: 'text',
      text: 'text' in part ? part.text : part.reason,
      citations:
        'citations' in part && part.citations
          ? part.citations
              .map((citation) => this.toCitationParam(citation))
              .filter((citation): citation is TextCitationParam => citation !== null)
          : undefined,
    };
  }

  private toCitationParam(citation: Citation): TextCitationParam | null {
    switch (citation.type) {
      case 'file_location':
      case 'others_location':
        return null;
      case 'web_location':
        return {
          type: 'web_search_result_location',
          cited_text: citation.citedText,
          title: citation.title ?? null,
          url: citation.url,
          encrypted_index: '',
        };
      default:
        return assertNever(citation as never);
    }
  }

  private toThinkingBlockParam(part: ThinkingPart): ThinkingBlockParam {
    return {
      type: 'thinking',
      thinking: part.content,
      signature: part.signature ?? '',
    };
  }

  private toImageBlockParam(part: ImagePart): ImageBlockParam | null {
    switch (part.image.sourceType) {
      case 'base64':
        switch (part.image.mimeType) {
          case 'image/jpeg':
          case 'image/png':
          case 'image/gif':
          case 'image/webp':
            return {
              type: 'image',
              source: {
                type: 'base64',
                media_type: part.image.mimeType,
                data: part.image.data,
              },
            };
          default:
            return null;
        }
      case 'url':
        return { type: 'image', source: { type: 'url', url: part.image.url } };
      case 'file_id':
        return null;
      default:
        return assertNever(part.image as never);
    }
  }

  private toDocumentBlockParam(part: DocumentPart): DocumentBlockParam | null {
    switch (part.document.sourceType) {
      case 'base64':
        switch (part.document.mimeType) {
          case 'text/plain':
            return {
              type: 'document',
              source: { type: 'text', media_type: 'text/plain', data: part.document.data! },
            };
          case 'application/pdf':
            return {
              type: 'document',
              source: { type: 'base64', media_type: 'application/pdf', data: part.document.data! },
            };
          default:
            return null;
        }
      case 'url':
        return { type: 'document', source: { type: 'url', url: part.document.url! } };
      case 'file_id':
        return null;
      default:
        return assertNever(part.document as never);
    }
  }

  private toToolUseBlockParam(part: ToolCallPart): ToolUseBlockParam | ServerToolUseBlockParam {
    if (part.providerSpecific && part.providerSpecific === ApiModes.ANTHROPIC) {
      return { ...JSON.parse(part.arguments), id: part.id, name: part.name } as ServerToolUseBlockParam;
    }

    return {
      type: 'tool_use',
      id: part.id,
      name: part.name,
      input: part.arguments === '' ? null : JSON.parse(part.arguments),
    } as ToolUseBlockParam;
  }

  private toToolResultBlockParam(part: ToolResultPart): ToolResultBlockParam {
    try {
      const parsedResult = !part.result || part.result === '' ? null : JSON.parse(part.result);

      if (part.providerSpecific === ApiModes.ANTHROPIC) {
        switch (part.name) {
          case 'bash_code_execution':
          case 'code_execution':
          case 'text_editor_code_execution':
          case 'tool_search':
          case 'web_fetch':
          case 'web_search': {
            const resultObject = typeof parsedResult === 'object' && parsedResult !== null ? parsedResult : {};
            return {
              ...(resultObject as Record<string, unknown>),
              type: `${part.name}_tool_result`,
              tool_use_id: part.id,
            } as unknown as ToolResultBlockParam;
          }
          default:
            break;
        }
      }

      if (parsedResult && Array.isArray(parsedResult)) {
        return {
          type: 'tool_result',
          tool_use_id: part.id,
          content: parsedResult.map((item) =>
            this.toBlockParam(item as unknown as ContentPart),
          ) as Array<TextBlockParam | ImageBlockParam | DocumentBlockParam>,
          is_error: part.isError ?? undefined,
        } as ToolResultBlockParam;
      }
    } catch {
      // Keep the fallback below for provider-specific or opaque tool result payloads.
    }

    return {
      type: 'tool_result',
      tool_use_id: part.id,
      content: part.result,
      is_error: part.isError ?? undefined,
    } as ToolResultBlockParam;
  }

  private toServerToolUseBlockParam(part: ServerToolUsePart): ToolUseBlockParam | ServerToolUseBlockParam | TextBlockParam {
    if (part.providerSpecific === ApiModes.ANTHROPIC) {
      try {
        return {
          ...(JSON.parse(part.data ?? '{}') as Record<string, unknown>),
          name: part.name,
        } as ServerToolUseBlockParam;
      } catch {
        // ignore and fall back to text placeholder
      }
    }

    return {
      type: 'text',
      text: `Tool use: ${part.name} with data ${part.data}`,
    };
  }

  private toAudioTextBlockParam(part: AudioPart): TextBlockParam {
    return {
      type: 'text',
      text: JSON.stringify({ audio: part.audio, type: part.type }),
    };
  }

  private toVideoTextBlockParam(part: VideoPart): TextBlockParam {
    return {
      type: 'text',
      text: JSON.stringify({ video: part.video, type: part.type }),
    };
  }
}

const rawConverter = new AnthropicRawMessageConverter();
const sessionConverter = new AnthropicSessionMessageConverter();
const blockPartConverter = new AnthropicBlockPartConverter();
const partConverter = new AnthropicPartConverter();

/**
 * Small public entry-point for Anthropic conversion logic.
 *
 * This keeps the raw/session conversion responsibilities isolated from the transport
 * implementation, and makes the extraction path easy to review before any stream logic is moved.
 */
export const AnthropicMessageConverter = {
  convertFromRawMessage(rawMsg: AnthropicApiMessage): SessionMessage {
    return rawConverter.convertFromRawMessage(rawMsg);
  },

  convertToRawMessage(msg: SessionMessage): AnthropicApiMessage {
    return sessionConverter.convertToRawMessage(msg);
  },

  toParts(block: ContentBlockParam | ContentBlock | ToolReferenceBlockParam): ContentPart[] {
    return blockPartConverter.toParts(block);
  },

  toBlockParam(part: ContentPart): ContentBlockParam | null {
    return partConverter.toBlockParam(part);
  },
};
