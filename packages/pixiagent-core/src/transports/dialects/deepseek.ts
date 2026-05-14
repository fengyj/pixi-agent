
import type {
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionStreamParams,
} from 'openai/resources/chat/completions';
import { ApiModes, SessionMessage, ThinkingPart, ContentPart } from '../../message';
import { ApiModeResolver, DialectResolver, ModelOptions } from '../base';
import type {
    MessageParam,
    MessageStreamParams,
    RawContentBlockDelta,
    ThinkingBlockParam,
} from '@anthropic-ai/sdk/resources/messages/messages';

export class DeepSeekApiModeResolver extends ApiModeResolver {
    getApiMode(model: string, baseUrl?: string): ApiModes | undefined {
        if (!baseUrl) return ApiModes.COMPLETIONS;
        else if (baseUrl.toLowerCase() === 'https://api.deepseek.com') return ApiModes.COMPLETIONS;
        else if (baseUrl.toLowerCase() === 'https://api.deepseek.com/anthropic') return ApiModes.ANTHROPIC;
        return undefined;
    }
    getBaseUrl(model: string, apiMode?: ApiModes): string | undefined {
        if (!apiMode || apiMode === ApiModes.COMPLETIONS)
            return 'https://api.deepseek.com';
        else if (apiMode === ApiModes.ANTHROPIC)
            return 'https://api.deepseek.com/anthropic';
        else
            return undefined;
    }
}

/**
 * Dialect resolver for DeepSeek via the OpenAI Chat Completions API (https://api.deepseek.com).
 * Handles the provider-specific `reasoning_content` field for thinking/reasoning.
 */
export class DeepSeekChatDialectResolver implements DialectResolver<ChatCompletionMessageParam, ChatCompletionChunk.Choice.Delta, ChatCompletionStreamParams> {

    // Models that doesn't support the reasoning_effort parameter
    private static readonly NON_REASONING_MODELS = ['deepseek-chat'];

    match(model: string, baseUrl: string): boolean {
        return baseUrl.toLowerCase() === 'https://api.deepseek.com';
    }

    manipulateOptions(options: ModelOptions, parameters: ChatCompletionStreamParams): ChatCompletionStreamParams {
        // DeepSeek doesn't support the 'developer' role; convert to 'system'
        parameters.messages = parameters.messages.map(msg =>
            msg.role === 'developer' ? { ...msg, role: 'system' } : msg
        ) as ChatCompletionMessageParam[];

        if (!options.thinkEffort) return parameters;
        if (DeepSeekChatDialectResolver.NON_REASONING_MODELS.includes(options.model)) {
            parameters.reasoning_effort = undefined;
            return parameters;
        }
        // DeepSeek reasoning_effort: 'none' | 'high' | 'max'
        const effortMap: Record<string, string | null> = {
            disable: 'none',
            low: 'high',
            medium: 'high',
            high: 'high',
            extreme: 'max',
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (parameters as any).extra_body = {
            'thinking': {
                'type': options.thinkEffort && options.thinkEffort === 'disable' ? 'disabled' : 'enabled'
            }
        };
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (parameters as any).reasoning_effort = effortMap[options.thinkEffort] ?? null;
        return parameters;
    }

    manipulateRawMessage(rawMsg: ChatCompletionMessageParam, msg?: SessionMessage): ChatCompletionMessageParam {
        if (!msg || rawMsg.role !== 'assistant' || !(msg.content instanceof Array)) return rawMsg;
        const thinkingParts = msg.content.filter((p): p is ThinkingPart => p.type === 'thinking');
        if (thinkingParts.length === 0) return rawMsg;
        // Write the accumulated thinking content back as reasoning_content
        (rawMsg as ChatCompletionAssistantMessageParam & { reasoning_content?: string }).reasoning_content =
            thinkingParts.map(p => p.content).join('');
        return rawMsg;
    }

    manipulateMessage(msg: SessionMessage, rawMsg: ChatCompletionMessageParam): SessionMessage {
        if (rawMsg.role !== 'assistant') return msg;
        const reasoningContent: string | undefined =
            (rawMsg as ChatCompletionAssistantMessageParam & { reasoning_content?: string }).reasoning_content;
        if (!reasoningContent) return msg;
        const thinkingPart: ThinkingPart = { type: 'thinking', content: reasoningContent };
        return { ...msg, content: ContentPart.concat([thinkingPart], msg.content) };
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extractFromDelta(data: string, delta: ChatCompletionChunk.Choice.Delta): any {
        if (data === 'reasoning') {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return (delta as any).reasoning_content ?? null;
        }
        return null;
    }
}

/**
 * Dialect resolver for DeepSeek via the Anthropic Messages API (https://api.deepseek.com/anthropic).
 * Handles thinking blocks in content using the Anthropic thinking block format.
 */
export class DeepSeekAnthropicDialectResolver implements DialectResolver<MessageParam, RawContentBlockDelta, MessageStreamParams> {

    match(model: string, baseUrl: string): boolean {
        return baseUrl.toLowerCase() === 'https://api.deepseek.com/anthropic';
    }

    // Models that support extended thinking
    private static readonly REASONING_MODELS = ['deepseek-reasoner'];

    manipulateOptions(options: ModelOptions, parameters: MessageStreamParams): MessageStreamParams {
        if (!options.thinkEffort || options.thinkEffort === 'disable') return parameters;
        if (!DeepSeekAnthropicDialectResolver.REASONING_MODELS.includes(options.model)) return parameters;
        // DeepSeek Anthropic uses output_config.effort for thinking strength.
        const effortMap: Record<string, 'high' | 'max'> = {
            low: 'high',
            medium: 'high',
            high: 'high',
            extreme: 'max',
        };
        const effort = effortMap[options.thinkEffort];
        if (effort) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (parameters as any).thinking = { type: 'enabled' };
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (parameters as any).output_config = { effort };
        }
        return parameters;
    }

    // Block types not supported by DeepSeek's Anthropic-compatible API.
    private static readonly UNSUPPORTED_BLOCK_TYPES = new Set([
        'image', 'document', 'search_result', 'server_tool_use',
        'web_search_tool_result', 'web_fetch_tool_result',
        'code_execution_tool_result', 'bash_code_execution_tool_result',
        'text_editor_code_execution_tool_result', 'tool_search_tool_result',
        'container_upload',
    ]);

    manipulateRawMessage(rawMsg: MessageParam, msg?: SessionMessage): MessageParam {
        // Step 1: filter out content block types that DeepSeek doesn't support (image, document, etc.).
        // Step 2: for assistant messages, inject unsigned thinking blocks (DeepSeek doesn't use signatures).
        //
        // We only clone rawMsg when we actually need to mutate it.
        // When msg is undefined, rawMsg wasn't produced from a SessionMessage conversion, so we must
        // clone before mutating to avoid side-effects on the original object.

        const rawContent = typeof rawMsg.content === 'string'
            ? [{ type: 'text' as const, text: rawMsg.content }]
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            : (rawMsg.content as any[]);

        // Collect unsupported block indices to know whether filtering is needed.
        const filteredContent = rawContent.filter(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (b: any) => !DeepSeekAnthropicDialectResolver.UNSUPPORTED_BLOCK_TYPES.has(b.type),
        );
        const needsFilter = filteredContent.length !== rawContent.length;

        // Collect unsigned thinking parts to inject (assistant only).
        const unsignedThinkingParts =
            rawMsg.role === 'assistant' && msg?.content instanceof Array
                ? msg.content.filter((p): p is ThinkingPart => p.type === 'thinking' && !p.signature)
                : [];
        const needsThinking = unsignedThinkingParts.length > 0;

        if (!needsFilter && !needsThinking) return rawMsg;

        // Clone only when we know a mutation is needed and msg is absent (shared reference risk).
        const base: MessageParam = msg ? rawMsg : structuredClone(rawMsg);

        const thinkingBlocks: ThinkingBlockParam[] = unsignedThinkingParts.map(p => ({
            type: 'thinking' as const,
            thinking: p.content,
            signature: '',
        }));

        return { ...base, content: [...thinkingBlocks, ...filteredContent] };
    }

    manipulateMessage(msg: SessionMessage, _rawMsg: MessageParam): SessionMessage {
        // Thinking blocks are already handled by AnthropicTransport.convertFromRawMessage.
        return msg;
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    extractFromDelta(_data: string, _delta: RawContentBlockDelta): any {
        // Not used: AnthropicTransport.generate handles all events natively via for-await.
        return null;
    }
}