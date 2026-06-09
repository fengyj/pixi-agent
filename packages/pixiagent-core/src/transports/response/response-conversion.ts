import type {
  EasyInputMessage,
  ResponseApplyPatchToolCall,
  ResponseApplyPatchToolCallOutput,
  ResponseCodeInterpreterToolCall,
  ResponseComputerToolCall,
  ResponseComputerToolCallOutputItem,
  ResponseCustomToolCall,
  ResponseCustomToolCallOutput,
  ResponseCustomToolCallOutputItem,
  ResponseFileSearchToolCall,
  ResponseFunctionShellToolCall,
  ResponseFunctionShellToolCallOutput,
  ResponseFunctionToolCall,
  ResponseFunctionToolCallOutputItem,
  ResponseFunctionWebSearch,
  ResponseInputFile,
  ResponseInputFileContent,
  ResponseInputImage,
  ResponseInputImageContent,
  ResponseInputItem,
  ResponseInputText,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseOutputRefusal,
  ResponseOutputText,
  ResponseReasoningItem,
  ResponseToolSearchCall,
  ResponseToolSearchOutputItem,
  ResponseToolSearchOutputItemParam,
} from 'openai/resources/responses/responses';
import {
  ApiModes,
  Citation,
  ContentPart,
  DocumentPart,
  ImagePart,
  RefusalPart,
  ServerToolUsePart,
  SessionMessage,
  TextPart,
  ThinkingPart,
  ToolCallPart,
  ToolResultPart,
} from '../../message';
import { assertNever } from '../../utils';

const ResponseContentPartHelper = {
  toContentPartsFromInputItem(item: ResponseInputItem): ContentPart[] {
    return ResponseContentPartHelper.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromOutputItem(item: ResponseOutputItem): ContentPart[] {
    return ResponseContentPartHelper.toContentPartsFromResponseItem(item);
  },

  toContentPartsFromResponseItem(item: ResponseInputItem | ResponseOutputItem): ContentPart[] {
    const isResponseMessageItem = (
      data: ResponseInputItem | ResponseOutputItem,
    ): data is EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage => {
      return (
        data.type === 'message' ||
        (!data.type &&
          'content' in data &&
          (typeof data.content === 'string' ||
            (Array.isArray(data.content) &&
              data.content.every(
                (c) =>
                  c.type === 'input_text' || c.type === 'input_image' || c.type === 'input_file',
              ))))
      );
    };

    if (isResponseMessageItem(item)) {
      return ResponseContentPartHelper.fromResponseMessageItem(item);
    }

    if (item.type === 'compaction' || item.type === 'compaction_trigger' || item.type === 'item_reference') {
      return [];
    }

    const part = ResponseContentPartHelper.fromResponseObject(item);
    return part ? [part] : [];
  },

  fromResponseMessageItem(
    item: EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage,
  ): ContentPart[] {
    if (typeof item.content === 'string') {
      return [{ type: 'text', text: item.content } as TextPart];
    }

    return item.content
      .map((c) => ResponseContentPartHelper.fromResponseObject(c))
      .filter((part): part is ContentPart => part !== null);
  },

  fromResponseObject(
    item:
      | Exclude<ResponseInputItem, EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage>
      | Exclude<ResponseOutputItem, ResponseOutputMessage>
      | ResponseOutputText
      | ResponseInputText
      | ResponseOutputRefusal
      | ResponseInputImage
      | ResponseInputImageContent
      | ResponseInputFile
      | ResponseInputFileContent,
  ): ContentPart | null {
    switch (item.type) {
      case 'input_text':
        return ResponseContentPartHelper.fromResponseInputText(item);
      case 'output_text':
        return ResponseContentPartHelper.fromResponseOutputText(item);
      case 'refusal':
        return ResponseContentPartHelper.fromResponseOutputRefusal(item);
      case 'input_image':
        return ResponseContentPartHelper.fromResponseInputImage(item);
      case 'input_file':
        return ResponseContentPartHelper.fromResponseInputFile(item);
      case 'image_generation_call':
        return ResponseContentPartHelper.fromResponseImageGenerationCall(item);
      case 'function_call':
        return ResponseContentPartHelper.fromResponseFunctionCall(item);
      case 'function_call_output':
        return ResponseContentPartHelper.fromResponseFunctionCallOutput(item);
      case 'reasoning':
        return ResponseContentPartHelper.fromResponseReasoningItem(item);
      case 'computer_call':
      case 'custom_tool_call':
      case 'tool_search_call':
      case 'local_shell_call':
      case 'shell_call':
      case 'apply_patch_call':
        return ResponseContentPartHelper.fromResponseSpecificToolCall(item);
      case 'computer_call_output':
      case 'custom_tool_call_output':
      case 'tool_search_output':
      case 'local_shell_call_output':
      case 'shell_call_output':
      case 'apply_patch_call_output':
        return ResponseContentPartHelper.fromResponseSpecificToolOutput(item);
      case 'file_search_call':
      case 'web_search_call':
      case 'code_interpreter_call':
      case 'mcp_call':
      case 'mcp_list_tools':
        return ResponseContentPartHelper.fromResponseServerToolCall(item);
      case 'mcp_approval_request':
      case 'mcp_approval_response':
      case 'compaction':
      case 'compaction_trigger':
        return null;
      case 'item_reference':
      case null:
      case undefined:
        return null;
      default:
        return assertNever(item);
    }
  },

  fromResponseInputText(item: ResponseInputText): TextPart {
    return { type: 'text', text: item.text };
  },

  fromResponseInputImage(item: ResponseInputImage | ResponseInputImageContent): ImagePart | null {
    if (item.image_url) {
      return {
        type: 'image',
        image: { sourceType: 'url', url: item.image_url },
      } as ImagePart;
    }
    if (item.file_id) {
      return {
        type: 'image',
        image: { sourceType: 'file_id', fileId: item.file_id },
      } as ImagePart;
    }
    return null;
  },

  fromResponseInputFile(item: ResponseInputFile | ResponseInputFileContent): DocumentPart | null {
    if (item.file_id) {
      return {
        type: 'document',
        document: { fileId: item.file_id, fileName: item.filename },
      } as DocumentPart;
    }
    if (item.file_url) {
      return {
        type: 'document',
        document: { url: item.file_url, fileName: item.filename },
      } as DocumentPart;
    }
    if (item.file_data) {
      return {
        type: 'document',
        document: {
          data: item.file_data,
          mimeType: 'text/plain',
          fileName: item.filename,
        },
      } as DocumentPart;
    }
    return null;
  },

  fromResponseOutputText(item: ResponseOutputText): TextPart {
    return {
      type: 'text',
      text: item.text,
      citations: item.annotations?.flatMap(ResponseContentPartHelper.fromAnnotation) ?? [],
    } as TextPart;
  },

  fromResponseOutputRefusal(item: ResponseOutputRefusal): RefusalPart {
    return { type: 'refusal', reason: item.refusal };
  },

  fromResponseImageGenerationCall(
    item: ResponseOutputItem.ImageGenerationCall | ResponseInputItem.ImageGenerationCall,
  ): ImagePart | null {
    if (!item.result) return null;
    return {
      type: 'image',
      image: {
        sourceType: 'base64',
        mimeType: 'image/png',
        data: item.result,
      },
    };
  },

  fromResponseFunctionCallOutput(item: ResponseInputItem.FunctionCallOutput): ToolResultPart {
    if (typeof item.output === 'string') {
      return { type: 'tool_result', id: item.call_id, result: item.output };
    }

    return {
      type: 'tool_result',
      id: item.call_id,
      result: JSON.stringify(
        item.output
          .map((c) => ResponseContentPartHelper.fromResponseObject(c))
          .filter((part): part is ContentPart => part !== undefined),
      ),
    };
  },

  fromResponseFunctionCall(item: ResponseFunctionToolCall): ToolCallPart {
    return {
      type: 'tool_call',
      id: item.call_id,
      name: item.name,
      arguments: item.arguments,
    };
  },

  fromResponseReasoningItem(item: ResponseReasoningItem): ThinkingPart | null {
    const summary = Array.isArray(item.summary)
      ? item.summary.map((summaryItem) => summaryItem.text).join('')
      : '';
    return summary ? { type: 'thinking', content: summary } : null;
  },

  fromResponseSpecificToolCall(
    item:
      | ResponseComputerToolCall
      | ResponseCustomToolCall
      | ResponseInputItem.ToolSearchCall
      | ResponseToolSearchCall
      | ResponseInputItem.LocalShellCall
      | ResponseOutputItem.LocalShellCall
      | ResponseInputItem.ShellCall
      | ResponseFunctionShellToolCall
      | ResponseInputItem.ApplyPatchCall
      | ResponseApplyPatchToolCall,
  ): ToolCallPart | ServerToolUsePart {
    if ('call_id' in item && item.call_id) {
      const { call_id, type, ...rest } = item;
      return {
        type: 'tool_call',
        id: call_id,
        name: type,
        arguments: JSON.stringify(rest),
        providerSpecific: ApiModes.RESPONSE,
      };
    }

    const { type, ...rest } = item;
    return {
      type: 'server_tool_use',
      name: type,
      data: JSON.stringify(rest),
      providerSpecific: ApiModes.RESPONSE,
    };
  },

  fromResponseSpecificToolOutput(
    item:
      | ResponseInputItem.ComputerCallOutput
      | ResponseComputerToolCallOutputItem
      | ResponseCustomToolCallOutput
      | ResponseCustomToolCallOutputItem
      | ResponseToolSearchOutputItemParam
      | ResponseToolSearchOutputItem
      | ResponseInputItem.LocalShellCallOutput
      | ResponseOutputItem.LocalShellCallOutput
      | ResponseInputItem.ShellCallOutput
      | ResponseFunctionShellToolCallOutput
      | ResponseInputItem.ApplyPatchCallOutput
      | ResponseApplyPatchToolCallOutput,
  ): ToolResultPart | ServerToolUsePart {
    if ('call_id' in item && item.call_id) {
      const { call_id, type, ...rest } = item;
      return {
        type: 'tool_result',
        id: call_id,
        name: type,
        result: JSON.stringify(rest),
        providerSpecific: ApiModes.RESPONSE,
      };
    }

    const { type, ...rest } = item;
    return {
      type: 'server_tool_use',
      name: type,
      data: JSON.stringify(rest),
      providerSpecific: ApiModes.RESPONSE,
    };
  },

  fromResponseServerToolCall(
    item:
      | ResponseFileSearchToolCall
      | ResponseFunctionWebSearch
      | ResponseCodeInterpreterToolCall
      | ResponseInputItem.McpCall
      | ResponseOutputItem.McpCall
      | ResponseInputItem.McpListTools
      | ResponseOutputItem.McpListTools,
  ): ServerToolUsePart {
    const { type, ...rest } = item;
    return {
      type: 'server_tool_use',
      name: type,
      data: JSON.stringify(rest),
      providerSpecific: ApiModes.RESPONSE,
    };
  },

  fromAnnotation(
    item:
      | ResponseOutputText.FileCitation
      | ResponseOutputText.URLCitation
      | ResponseOutputText.ContainerFileCitation
      | ResponseOutputText.FilePath,
  ): Citation {
    switch (item.type) {
      case 'file_citation':
        return {
          type: 'file_location',
          fileId: item.file_id,
          fileName: item.filename,
          citedText: '',
          extra: { index: item.index },
        } as Citation;
      case 'url_citation':
        return {
          type: 'web_location',
          url: item.url,
          title: item.title,
          citedText: '',
          startIndex: item.start_index,
          endIndex: item.end_index,
        } as Citation;
      case 'container_file_citation':
        return {
          type: 'file_location',
          fileId: item.file_id,
          fileName: item.filename,
          citedText: '',
          startIndex: item.start_index,
          endIndex: item.end_index,
          extra: { container_id: item.container_id },
        } as Citation;
      case 'file_path':
        return {
          type: 'file_location',
          fileId: item.file_id,
          citedText: '',
          fileName: '',
          extra: { index: item.index },
        } as Citation;
      default:
        return assertNever(item);
    }
  },
};

const ContentPartResponseHelper = {
  normalizeSessionMessage(
    message: SessionMessage | string,
    content?: Array<ContentPart>,
  ): SessionMessage {
    if (typeof message === 'string') {
      return {
        messageId: `tmp_${message}`,
        type: 'session_message',
        role: message as 'assistant' | 'user' | 'tool',
        content: content ?? [],
      };
    }
    return message;
  },

  toResponseItems(
    message: SessionMessage | string,
    content?: Array<ContentPart>,
  ): Array<ResponseInputItem | ResponseOutputItem> {
    const normalized = ContentPartResponseHelper.normalizeSessionMessage(message, content);
    if (normalized.role === 'assistant') {
      return ContentPartResponseHelper.toResponseOutputItems(
        normalized as Extract<SessionMessage, { role: 'assistant' }>,
      );
    }

    return ContentPartResponseHelper.toResponseInputItems(
      normalized as Extract<SessionMessage, { role: 'user' | 'tool' }>,
    );
  },

  toResponseInputItems(
    message: Omit<SessionMessage, 'role'> & { role: 'user' | 'tool' },
  ): ResponseInputItem[] {
    if (typeof message.content === 'string') {
      return [ContentPartResponseHelper.toInputMessage(message.content)];
    }

    const items: Array<
      | Exclude<ResponseInputItem, ResponseOutputMessage | EasyInputMessage | ResponseInputItem.Message>
      | ResponseInputText
      | ResponseInputImage
      | ResponseInputFile
    > = [];

    for (const part of message.content) {
      switch (part.type) {
        case 'text':
          items.push(ContentPartResponseHelper.toInputText(part));
          break;
        case 'image':
          items.push(ContentPartResponseHelper.toInputImage(part));
          break;
        case 'document':
          items.push(ContentPartResponseHelper.toInputFile(part));
          break;
        case 'tool_call':
          items.push(ContentPartResponseHelper.toToolCall(part as ToolCallPart));
          break;
        case 'tool_result':
          items.push(ContentPartResponseHelper.toToolResult(part as ToolResultPart));
          break;
        case 'server_tool_use':
          items.push(
            ContentPartResponseHelper.toItemFromServerToolUseForInput(part as ServerToolUsePart),
          );
          break;
        case 'audio':
        case 'video':
        case 'refusal':
        case 'thinking':
          break;
        default:
          assertNever(part);
      }
    }

    const itemsForMessage: Array<ResponseInputText | ResponseInputImage | ResponseInputFile> = [];
    const inputItems: Array<ResponseInputItem> = [];

    const flush = (): void => {
      if (itemsForMessage.length > 0) {
        inputItems.push(ContentPartResponseHelper.toInputMessage([...itemsForMessage]));
        itemsForMessage.length = 0;
      }
    };

    for (const item of items) {
      switch (item.type) {
        case 'input_text':
        case 'input_image':
        case 'input_file':
          itemsForMessage.push(item);
          break;
        default:
          flush();
          inputItems.push(item);
      }
    }

    flush();
    return inputItems;
  },

  toResponseOutputItems(
    message: Omit<SessionMessage, 'role'> & { role: 'assistant' },
  ): ResponseOutputItem[] {
    if (typeof message.content === 'string') {
      return [ContentPartResponseHelper.toOutputMessage(message.messageId, message.content)];
    }

    const items: Array<Exclude<ResponseOutputItem, ResponseOutputMessage> | ResponseOutputText | ResponseOutputRefusal> = [];

    for (const part of message.content) {
      switch (part.type) {
        case 'text':
          items.push(ContentPartResponseHelper.toOutputText(part));
          break;
        case 'refusal':
          items.push(ContentPartResponseHelper.toRefusal(part));
          break;
        case 'thinking':
          items.push(ContentPartResponseHelper.toThinking(part));
          break;
        case 'image':
          if (part.image.sourceType === 'base64') {
            items.push(ContentPartResponseHelper.toImageGenerationCall(part as ImagePart & { image: { sourceType: 'base64' } }));
          }
          break;
        case 'tool_call':
          items.push(ContentPartResponseHelper.toToolCall(part as ToolCallPart));
          break;
        case 'tool_result':
          items.push(ContentPartResponseHelper.toToolResultForOutputItem(part as ToolResultPart));
          break;
        case 'server_tool_use':
          items.push(ContentPartResponseHelper.toItemFromServerToolUse(part as ServerToolUsePart));
          break;
        case 'audio':
        case 'document':
        case 'video':
          break;
        default:
          assertNever(part);
      }
    }

    const itemsForMessage: Array<ResponseOutputText | ResponseOutputRefusal> = [];
    const itemsForThinking: Array<ResponseReasoningItem> = [];
    const outputItems: Array<ResponseOutputItem> = [];

    const flushMessage = (): void => {
      if (itemsForMessage.length > 0) {
        outputItems.push(
          ContentPartResponseHelper.toOutputMessage(
            undefined,
            [...itemsForMessage],
          ),
        );
        itemsForMessage.length = 0;
      }
    };

    const flushThinking = (): void => {
      if (itemsForThinking.length > 0) {
        outputItems.push({
          id: `${message.messageId}_${outputItems.length}`,
          type: 'reasoning',
          summary: itemsForThinking.flatMap((item) => item.summary),
        } as ResponseReasoningItem);
        itemsForThinking.length = 0;
      }
    };

    for (const item of items) {
      switch (item.type) {
        case 'output_text':
        case 'refusal':
          flushThinking();
          itemsForMessage.push(item);
          break;
        case 'reasoning':
          flushMessage();
          itemsForThinking.push(item);
          break;
        default:
          flushThinking();
          flushMessage();
          outputItems.push(item);
      }
    }

    flushThinking();
    flushMessage();
    return outputItems;
  },

  toOutputMessage(
    id: string | undefined,
    items: string | Array<ResponseOutputText | ResponseOutputRefusal>,
  ): ResponseOutputMessage {
    const base = {
      type: 'message' as const,
      role: 'assistant' as const,
      content: typeof items === 'string'
        ? [{ type: 'output_text' as const, text: items, annotations: [] as never[] }]
        : items,
      status: 'completed' as const,
    };

    return (id ? { ...base, id } : base) as ResponseOutputMessage;
  },

  toInputMessage(content: string | Array<ResponseInputText | ResponseInputImage | ResponseInputFile>): EasyInputMessage | ResponseInputItem.Message {
    if (typeof content === 'string') {
      return { type: 'message', role: 'user', content };
    }

    return { type: 'message', role: 'user', content };
  },

  toOutputText(part: TextPart): ResponseOutputText {
    return {
      type: 'output_text',
      text: part.text,
      annotations: (part.citations ?? [])
        .map((citation) => ContentPartResponseHelper.toAnnotation(citation))
        .filter((annotation): annotation is NonNullable<typeof annotation> => annotation !== null),
    };
  },

  toRefusal(part: RefusalPart): ResponseOutputRefusal {
    return { type: 'refusal', refusal: part.reason };
  },

  toItemFromServerToolUse(
    part: ServerToolUsePart,
  ): Exclude<ResponseOutputItem, ResponseOutputMessage> | ResponseOutputText | ResponseOutputRefusal {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      try {
        const parsed = JSON.parse(part.data ?? '{}') as object;
        return {
          ...parsed,
          type: part.name,
        } as Exclude<ResponseOutputItem, ResponseOutputMessage> | ResponseOutputText | ResponseOutputRefusal;
      } catch {
        // Ignore invalid tool payloads and fall back.
      }
    }

    return {
      type: 'output_text',
      text: `Tool use: ${part.name} with data ${part.data}`,
      annotations: [],
    } as ResponseOutputText;
  },

  toItemFromServerToolUseForInput(
    part: ServerToolUsePart,
  ): Exclude<ResponseInputItem, ResponseOutputMessage | EasyInputMessage | ResponseInputItem.Message> | ResponseInputText | ResponseInputImage | ResponseInputFile {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      try {
        const parsed = JSON.parse(part.data ?? '{}') as Record<string, unknown>;
        return {
          ...(parsed as object),
          type: part.name,
          call_id: (parsed as { id?: string }).id ?? part.name,
        } as Exclude<ResponseInputItem, ResponseOutputMessage | EasyInputMessage | ResponseInputItem.Message> | ResponseInputText | ResponseInputImage | ResponseInputFile;
      } catch {
        // Ignore invalid tool payloads and fall back.
      }
    }

    return {
      type: 'input_text',
      text: `Tool use: ${part.name} with data ${part.data}`,
    } as ResponseInputText;
  },

  toImageGenerationCall(part: ImagePart & { image: { sourceType: 'base64' } }): ResponseOutputItem.ImageGenerationCall {
    return {
      type: 'image_generation_call',
      id: '',
      result: part.image.data,
      status: part.image.data === '' ? 'failed' : 'completed',
    } as ResponseOutputItem.ImageGenerationCall;
  },

  toToolCall(part: ToolCallPart): ResponseFunctionToolCall {
    if (part.providerSpecific === ApiModes.RESPONSE) {
      return {
        ...JSON.parse(part.arguments),
        type: part.name,
        call_id: part.id,
        status: 'completed',
      } as ResponseFunctionToolCall;
    }

    return {
      type: 'function_call',
      id: part.id,
      call_id: part.id,
      name: part.name,
      arguments: part.arguments,
      status: 'completed',
    } as ResponseFunctionToolCall;
  },

  toToolResult(part: ToolResultPart): ResponseInputItem.FunctionCallOutput | ResponseInputItem.ComputerCallOutput | ResponseCustomToolCallOutput | ResponseToolSearchOutputItemParam | ResponseInputItem.LocalShellCallOutput | ResponseInputItem.ShellCallOutput | ResponseInputItem.ApplyPatchCallOutput {
    try {
      const parsedResult = !part.result || part.result === '' ? null : JSON.parse(part.result);

      if (part.providerSpecific === ApiModes.RESPONSE) {
        return { ...parsedResult, type: part.name, call_id: part.id };
      }

      if (parsedResult && Array.isArray(parsedResult)) {
        const convertable = parsedResult.every((candidate) => {
          if (!('type' in candidate)) return false;
          switch (candidate.type) {
            case 'text':
              return 'text' in candidate && typeof candidate.text === 'string';
            case 'image':
              return (
                ('image_url' in candidate && typeof candidate.image_url === 'string') ||
                ('file_id' in candidate && typeof candidate.file_id === 'string')
              );
            case 'document':
              return (
                ('file_url' in candidate && typeof candidate.file_url === 'string') ||
                ('file_data' in candidate && typeof candidate.file_data === 'string') ||
                ('file_id' in candidate && typeof candidate.file_id === 'string')
              );
            default:
              return false;
          }
        });

        if (convertable) {
          const outputParts = parsedResult.map((candidate) => candidate as TextPart | ImagePart | DocumentPart);
          return {
            type: 'function_call_output',
            call_id: part.id,
            output: outputParts
              .map((candidate) => {
                switch (candidate.type) {
                  case 'text':
                    return ContentPartResponseHelper.toInputText(candidate);
                  case 'image':
                    return ContentPartResponseHelper.toInputImage(candidate);
                  case 'document':
                    return ContentPartResponseHelper.toInputFile(candidate);
                  default:
                    return assertNever(candidate);
                }
              })
              .filter((item) => item !== null) as Array<ResponseInputText | ResponseInputImage>,
            status: 'completed',
          };
        }
      }
    } catch {
      // Ignore invalid JSON payloads, fall back to string output.
    }

    return {
      type: 'function_call_output',
      call_id: part.id,
      output: part.result ?? '',
      status: 'completed',
    };
  },

  toToolResultForOutputItem(part: ToolResultPart): ResponseFunctionToolCallOutputItem | ResponseComputerToolCallOutputItem | ResponseCustomToolCallOutputItem | ResponseToolSearchOutputItem | ResponseOutputItem.LocalShellCallOutput | ResponseFunctionShellToolCallOutput | ResponseApplyPatchToolCallOutput {
    const { id, ...rest } = ContentPartResponseHelper.toToolResult(part);

    switch (rest.type) {
      case 'computer_call_output':
        return { ...rest, id: id ?? '', status: rest.status ?? 'completed', acknowledged_safety_checks: rest.acknowledged_safety_checks ?? undefined };
      case 'apply_patch_call_output':
        return { ...rest, id: id ?? '' };
      case 'local_shell_call_output':
        return { ...rest, id: id ?? '' };
      case 'shell_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          max_output_length: rest.max_output_length ?? null,
        };
      case 'tool_search_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          call_id: rest.call_id ?? '',
          execution: rest.execution ?? 'server',
        };
      case 'function_call_output':
        return {
          ...rest,
          id: id ?? '',
          status: rest.status && rest.status !== 'completed' ? 'incomplete' : 'completed',
          output:
            typeof rest.output === 'string'
              ? rest.output
              : rest.output.map((candidate) => {
                  switch (candidate.type) {
                    case 'input_text':
                      return candidate as ResponseInputText;
                    case 'input_image':
                      return candidate as ResponseInputImage;
                    case 'input_file':
                      return candidate as ResponseInputFile;
                    default:
                      return assertNever(candidate);
                  }
                }),
        };
      case 'custom_tool_call_output':
        return { ...rest, id: id ?? '', status: 'completed' };
      default:
        return assertNever(rest);
    }
  },

  toThinking(part: ThinkingPart): ResponseReasoningItem {
    return {
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: part.content }],
      id: '',
    };
  },

  toInputText(part: TextPart): ResponseInputText {
    return { type: 'input_text', text: part.text };
  },

  toInputImage(part: ImagePart): ResponseInputImage {
    switch (part.image.sourceType) {
      case 'base64':
        return {
          type: 'input_image',
          detail: 'auto',
          image_url: `data:${part.image.mimeType};base64,${part.image.data}`,
        };
      case 'file_id':
        return { type: 'input_image', detail: 'auto', file_id: part.image.fileId };
      case 'url':
        return { type: 'input_image', detail: 'auto', image_url: part.image.url };
      default:
        return assertNever(part.image);
    }
  },

  toInputFile(part: DocumentPart): ResponseInputFile {
    const source = part.document;
    switch (source.sourceType) {
      case 'url':
        return { type: 'input_file', file_url: source.url, filename: source.fileName };
      case 'base64':
        return { type: 'input_file', file_data: source.data, filename: source.fileName };
      case 'file_id':
        return { type: 'input_file', file_id: source.fileId, filename: source.fileName };
      default:
        return assertNever(source);
    }
  },

  toAnnotation(citation: Citation):
    | ResponseOutputText.FileCitation
    | ResponseOutputText.URLCitation
    | ResponseOutputText.ContainerFileCitation
    | ResponseOutputText.FilePath
    | null {
    switch (citation.type) {
      case 'web_location':
        return {
          type: 'url_citation',
          url: citation.url,
          title: citation.title ?? '',
          start_index: citation.startIndex ?? 0,
          end_index: citation.endIndex ?? 0,
        };
      case 'file_location': {
        const fileLocation = citation as Citation & {
          extra?: { file_id?: string; container_id?: string; index?: number };
        };
        const fileId = fileLocation.extra?.file_id;
        const filename = citation.fileName;
        const container_id = fileLocation.extra?.container_id;
        if (!fileId || typeof fileId !== 'string') return null;
        if (container_id !== undefined && typeof container_id === 'string') {
          return {
            type: 'container_file_citation',
            file_id: fileId,
            filename,
            start_index: citation.startIndex ?? 0,
            end_index: citation.endIndex ?? 0,
            container_id,
          };
        }
        if (filename === '') {
          return {
            type: 'file_path',
            file_id: fileId,
            index: typeof fileLocation.extra?.index === 'number' ? fileLocation.extra.index : 0,
          };
        }
        return {
          type: 'file_citation',
          file_id: fileId,
          filename,
          index: typeof fileLocation.extra?.index === 'number' ? fileLocation.extra.index : 0,
        };
      }
      case 'others_location':
        return null;
      default:
        return assertNever(citation);
    }
  },
};

export const ResponseConversionHelper = {
  toContentParts: ResponseContentPartHelper.toContentPartsFromResponseItem,
  toContentPart: ResponseContentPartHelper.fromResponseObject,
  toResponseInputItems: ContentPartResponseHelper.toResponseInputItems,
  toResponseOutputItems: ContentPartResponseHelper.toResponseOutputItems,
  toResponseItems: ContentPartResponseHelper.toResponseItems,
};
