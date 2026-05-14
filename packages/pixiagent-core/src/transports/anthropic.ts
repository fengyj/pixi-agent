import Anthropic from '@anthropic-ai/sdk';
import type {
  MessageParam,
  Tool as AnthropicTool,
  TextBlockParam,
  ImageBlockParam,
  DocumentBlockParam,
  ToolUseBlockParam,
  ToolResultBlockParam,
  ThinkingBlockParam,
  RawContentBlockDelta,
} from '@anthropic-ai/sdk/resources/messages';
import type { MessageStreamParams } from '@anthropic-ai/sdk/resources/messages/messages';
import { ProviderTransport, ModelOptions, StreamCallbacks, DialectResolver, ModelRequestOptions } from './base';
import {
  ApiModes,
  SessionMessage,
  InternalMessage,
  ContentPart,
  TextPart,
  ThinkingPart,
  ToolCallPart,
  ToolResultPart,
  ImagePart,
  DocumentPart,
  UsageStats,
} from '../message';

export class AnthropicTransport extends ProviderTransport<MessageParam> {
  readonly client: Anthropic;

  constructor(
    baseUrl?: string,
    apiKey?: string,
    dialectResolver?: DialectResolver<MessageParam, RawContentBlockDelta, MessageStreamParams>,
  ) {
    super(ApiModes.ANTHROPIC, dialectResolver);
    this.client = new Anthropic({ baseURL: baseUrl, apiKey });
  }

  // ─── convertFromRawMessage ────────────────────────────────────────────────

  private getFromAssistantMessageParam(rawMsg: MessageParam): SessionMessage {
    if (typeof rawMsg.content === 'string') {
      return { type: 'session_message', role: 'assistant', content: rawMsg.content };
    }
    const parts: ContentPart[] = [];
    for (const block of rawMsg.content) {
      if (block.type === 'text') {
        parts.push({ type: 'text', text: block.text } as TextPart);
      } else if (block.type === 'thinking') {
        parts.push({
          type: 'thinking',
          content: block.thinking,
          signature: block.signature,
        } as ThinkingPart);
      } else if (block.type === 'tool_use') {
        parts.push({
          type: 'tool_call',
          id: block.id,
          name: block.name,
          arguments: JSON.stringify(block.input),
        } as ToolCallPart);
      }
      // skip redacted_thinking and other unrecognised block types
    }
    return { type: 'session_message', role: 'assistant', content: parts };
  }

  private getFromUserMessageParam(rawMsg: MessageParam): SessionMessage {
    if (typeof rawMsg.content === 'string') {
      return { type: 'session_message', role: 'user', content: rawMsg.content };
    }
    const parts: ContentPart[] = [];
    let hasToolResult = false;
    let hasOtherContent = false;
    for (const block of rawMsg.content) {
      if (block.type === 'text') {
        parts.push({ type: 'text', text: block.text } as TextPart);
        hasOtherContent = true;
      } else if (block.type === 'image') {
        const src = block.source;
        if (src.type === 'base64') {
          parts.push({
            type: 'image',
            image: { sourceType: 'base64', mimeType: src.media_type, data: src.data },
          } as ImagePart);
        } else if (src.type === 'url') {
          parts.push({ type: 'image', image: { sourceType: 'url', url: src.url } } as ImagePart);
        }
        hasOtherContent = true;
      } else if (block.type === 'document') {
        const src = block.source;
        if (src.type === 'base64') {
          parts.push({
            type: 'document',
            document: { sourceType: 'base64', mimeType: src.media_type, data: src.data },
          } as DocumentPart);
        } else if (src.type === 'text') {
          parts.push({
            type: 'document',
            document: { sourceType: 'base64', mimeType: src.media_type, data: src.data },
          } as DocumentPart);
        } else if (src.type === 'url') {
          parts.push({
            type: 'document',
            document: { sourceType: 'url', url: src.url },
          } as DocumentPart);
        }
        // 'content' source type has no direct equivalent in DocumentPart — skip
        hasOtherContent = true;
      } else if (block.type === 'tool_result') {
        const content = block.content;
        let result = '';
        if (typeof content === 'string') {
          result = content;
        } else if (Array.isArray(content)) {
          const textBlock = content.find((b) => b.type === 'text');
          if (textBlock && textBlock.type === 'text') result = textBlock.text;
        }
        parts.push({
          type: 'tool_result',
          id: block.tool_use_id,
          result,
          isError: block.is_error ?? undefined,
        } as ToolResultPart);
        hasToolResult = true;
      }
    }
    return {
      type: 'session_message',
      role: hasToolResult && !hasOtherContent ? 'tool' : 'user',
      content: parts,
    };
  }

  convertFromRawMessage(rawMsg: MessageParam): SessionMessage {
    const msg = (() => {
      switch (rawMsg.role) {
        case 'assistant':
          return this.getFromAssistantMessageParam(rawMsg);
        case 'user':
          return this.getFromUserMessageParam(rawMsg);
        default:
          throw new Error(`Unsupported message role: ${rawMsg.role}`);
      }
    })();
    return this.dialectResolver ? this.dialectResolver.manipulateMessage(msg, rawMsg) : msg;
  }

  // ─── convertToRawMessage (pure — dialect is applied in buildStreamParams) ─

  private getAssistantMessageParam(msg: SessionMessage): MessageParam {
    if (!msg.content) return { role: 'assistant', content: '' };
    if (typeof msg.content === 'string') return { role: 'assistant', content: msg.content };
    const blocks: (TextBlockParam | ThinkingBlockParam | ToolUseBlockParam)[] = [];
    for (const part of msg.content) {
      if (part.type === 'text') {
        blocks.push({ type: 'text', text: (part as TextPart).text });
      } else if (part.type === 'thinking') {
        const tp = part as ThinkingPart;
        // Only include thinking blocks that carry a signature (required by the Anthropic API).
        // Dialect resolvers (e.g. DeepSeek) may inject unsigned thinking blocks via manipulateRawMessage.
        if (tp.signature) {
          blocks.push({ type: 'thinking', thinking: tp.content, signature: tp.signature });
        }
      } else if (part.type === 'tool_call') {
        const tc = part as ToolCallPart;
        blocks.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: JSON.parse(tc.arguments || '{}'),
        });
      }
    }
    return { role: 'assistant', content: blocks };
  }

  private getUserMessageParam(msg: SessionMessage): MessageParam {
    if (!msg.content || (typeof msg.content !== 'string' && msg.content.length === 0)) {
      throw new Error('User message must have non-empty content.');
    }
    if (typeof msg.content === 'string') return { role: 'user', content: msg.content };
    const blocks: (TextBlockParam | ImageBlockParam | DocumentBlockParam)[] = [];
    for (const part of msg.content) {
      if (part.type === 'text') {
        blocks.push({ type: 'text', text: (part as TextPart).text });
      } else if (part.type === 'image') {
        const img = (part as ImagePart).image;
        if (img.sourceType === 'base64') {
          blocks.push({
            type: 'image',
            source: {
              type: 'base64',
              media_type: img.mimeType as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
              data: img.data,
            },
          });
        } else if (img.sourceType === 'url') {
          blocks.push({ type: 'image', source: { type: 'url', url: img.url } });
        }
      } else if (part.type === 'document') {
        const doc = (part as DocumentPart).document;
        if (doc.sourceType === 'base64') {
          if (doc.mimeType === 'text/plain') {
            blocks.push({
              type: 'document',
              source: { type: 'text', media_type: 'text/plain', data: doc.data! },
            });
          } else {
            blocks.push({
              type: 'document',
              source: { type: 'base64', media_type: 'application/pdf', data: doc.data! },
            });
          }
        } else if (doc.sourceType === 'url') {
          blocks.push({ type: 'document', source: { type: 'url', url: doc.url! } });
        }
        // file_id not supported by this SDK version — skip
      }
    }
    return { role: 'user', content: blocks };
  }

  private getToolMessageParam(msg: SessionMessage): MessageParam {
    if (!msg.content || typeof msg.content === 'string' || msg.content.length === 0) {
      throw new Error('Tool message content must be a non-empty array of content parts.');
    }
    const blocks: ToolResultBlockParam[] = msg.content
      .filter((p) => p.type === 'tool_result')
      .map((p) => p as ToolResultPart)
      .map((p) => ({
        type: 'tool_result' as const,
        tool_use_id: p.id,
        content: p.result,
        is_error: p.isError,
      }));
    return { role: 'user', content: blocks };
  }

  convertToRawMessage(msg: SessionMessage): MessageParam {
    const rawMsg = (() => {
      switch (msg.role) {
        case 'assistant':
          return this.getAssistantMessageParam(msg);
        case 'user':
          return this.getUserMessageParam(msg);
        case 'tool':
          return this.getToolMessageParam(msg);
        default:
          throw new Error(`Unsupported message role: ${msg.role}`);
      }
    })();
    return this.dialectResolver ? this.dialectResolver.manipulateRawMessage(rawMsg, msg) : rawMsg;
  }

  // ─── helpers ─────────────────────────────────────────────────────────────

  private getThinkingConfig(
    thinkEffort?: ModelOptions['thinkEffort'],
  ): Anthropic.Messages.ThinkingConfigParam | undefined {
    switch (thinkEffort) {
      case 'disable':
        return { type: 'disabled' };
      case 'low':
        return { type: 'enabled', budget_tokens: 1024 };
      case 'medium':
        return { type: 'enabled', budget_tokens: 4096 };
      case 'high':
        return { type: 'enabled', budget_tokens: 16000 };
      case 'extreme':
        return { type: 'enabled', budget_tokens: 32000 };
      default:
        return undefined;
    }
  }

  private getToolChoice(options: ModelOptions): Anthropic.Messages.ToolChoice | undefined {
    if (!options.tools || options.tools.length === 0) return undefined;
    const disableParallel = options.parallelToolCalls === false ? true : undefined;
    if (options.toolChoice === 'force')
      return { type: 'any', disable_parallel_tool_use: disableParallel };
    if (options.toolChoice === 'none') return { type: 'none' };
    if (disableParallel) return { type: 'auto', disable_parallel_tool_use: true };
    return undefined;
  }

  /**
   * Merge consecutive messages with the same role into one.
   * The Anthropic API requires strict user/assistant alternation.
   */
  private mergeConsecutiveSameRoleMessages(messages: MessageParam[]): MessageParam[] {
    const result: MessageParam[] = [];
    for (const msg of messages) {
      const last = result[result.length - 1];
      if (last && last.role === msg.role) {
        const a =
          typeof last.content === 'string'
            ? [{ type: 'text' as const, text: last.content }]
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            : [...(last.content as any[])];
        const b =
          typeof msg.content === 'string'
            ? [{ type: 'text' as const, text: msg.content }]
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            : [...(msg.content as any[])];
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        last.content = [...a, ...b] as any;
      } else {
        result.push({ ...msg });
      }
    }
    return result;
  }

  private buildStreamParams(
    options: ModelOptions,
    messages: Array<MessageParam>,
  ): MessageStreamParams {

    const mergedMessages = this.mergeConsecutiveSameRoleMessages(messages);

    const tools: AnthropicTool[] | undefined = options.tools?.map(
      (t) =>
        ({
          name: t.name,
          description: t.description,
          input_schema: t.parameters,
        }) as AnthropicTool,
    );

    const params: MessageStreamParams = {
      model: options.model,
      messages: mergedMessages,
      system: options.systemPrompt,
      max_tokens: options.maxTokens ?? 4096,
      temperature: options.temperature,
      stop_sequences: options.stopSequences,
      tools: tools && tools.length > 0 ? tools : undefined,
      tool_choice: this.getToolChoice(options),
      thinking: this.getThinkingConfig(options.thinkEffort),
      metadata: options.metadata?.['user_id']
        ? { user_id: String(options.metadata['user_id']) }
        : undefined,
    };

    return this.dialectResolver ? this.dialectResolver.manipulateOptions(options, params) : params;
  }

  async generate(
    options: ModelOptions,
    messages: Array<MessageParam>,
    callbacks?: StreamCallbacks,
    requestOptions?: ModelRequestOptions,
  ): Promise<{ rawMessageId: string, rawMessage: MessageParam; usage?: UsageStats }> {
    const params = this.buildStreamParams(options, messages);

    let textChunkStarted = false;
    let thinkingChunkStarted = false;
    let thinkingText = '';
    let currentToolName: string | undefined;

    const stream = this.client.messages.stream(params, requestOptions);

    try {
      for await (const event of stream) {
        if (event.type === 'content_block_start') {
          if (event.content_block.type === 'tool_use') {
            currentToolName = event.content_block.name;
          }
        } else if (event.type === 'content_block_delta') {
          const delta = event.delta;
          if (delta.type === 'text_delta') {
            if (thinkingChunkStarted) {
              thinkingChunkStarted = false;
              callbacks?.onThinkingChunk?.('', 'end');
              callbacks?.onThinking?.(thinkingText);
              thinkingText = '';
            }
            if (!textChunkStarted) {
              textChunkStarted = true;
              callbacks?.onTextChunk?.(delta.text, 'begin');
            } else {
              callbacks?.onTextChunk?.(delta.text);
            }
            callbacks?.onText?.(delta.text);
          } else if (delta.type === 'thinking_delta') {
            thinkingText += delta.thinking;
            if (!thinkingChunkStarted) {
              thinkingChunkStarted = true;
              callbacks?.onThinkingChunk?.(delta.thinking, 'begin');
            } else {
              callbacks?.onThinkingChunk?.(delta.thinking);
            }
          } else if (delta.type === 'input_json_delta' && currentToolName) {
            callbacks?.onToolUse?.(currentToolName, delta.partial_json);
          }
        } else if (event.type === 'content_block_stop') {
          currentToolName = undefined;
        } else if (event.type === 'message_stop') {
          if (textChunkStarted) {
            textChunkStarted = false;
            callbacks?.onTextChunk?.('', 'end');
          }
          if (thinkingChunkStarted) {
            thinkingChunkStarted = false;
            callbacks?.onThinkingChunk?.('', 'end');
            callbacks?.onThinking?.(thinkingText);
            thinkingText = '';
          }
        }
      }

      // Safety net: flush open chunks if message_stop was not observed
      if (textChunkStarted) callbacks?.onTextChunk?.('', 'end');
      if (thinkingChunkStarted) {
        callbacks?.onThinkingChunk?.('', 'end');
        callbacks?.onThinking?.(thinkingText);
      }
    } catch (error) {
      await callbacks?.onError?.(error as Error);
      throw error;
    }

    const response = await stream.finalMessage();

    return {
      rawMessageId: response.id,
      rawMessage: { role: 'assistant', content: response.content } as unknown as MessageParam,
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens,
      }
    };
  }
}
