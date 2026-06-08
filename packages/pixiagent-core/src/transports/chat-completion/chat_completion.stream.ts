import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageFunctionToolCall,
} from 'openai/resources/chat/completions';
import type { DialectResolver, StreamCallbacks } from '../base';
import { StreamDataExtractor } from '../base';
import type { ChatCompletionApiMessage } from '../../message';

export class ChatCompletionStreamProcessor {
  constructor(
    private readonly dialectResolver?: DialectResolver<
      ChatCompletionApiMessage,
      ChatCompletionChunk.Choice.Delta,
      never,
      ChatCompletion
    >,
  ) {}

  async process(
    stream: AsyncIterable<ChatCompletionChunk>,
    callbacks?: StreamCallbacks,
  ): Promise<ChatCompletion> {
    const streamDataExtractor = new StreamDataExtractor(
      {
      object: 'chat.completion',
        id: '',
        model: '',
        created: Date.now() / 1000,
        usage: undefined,
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            logprobs: null,
            message: {
              role: 'assistant',
              content: null as string | null,
              tool_calls: undefined as Array<ChatCompletionMessageFunctionToolCall> | undefined,
              audio: null,
            },
          },
        ],
      } as ChatCompletion,
      callbacks,
    );

    for await (const chunk of stream) {
      await this.applyChunk(chunk, streamDataExtractor);
    }

    return streamDataExtractor.accumulatedData;
  }

  private async applyChunk(
    chunk: ChatCompletionChunk,
    streamDataExtractor: StreamDataExtractor<ChatCompletion>,
  ): Promise<void> {
    if (chunk.choices.length === 0) {
      return;
    }
    if (chunk.usage) {
      streamDataExtractor.accumulatedData.usage = chunk.usage;
    }
    if (chunk.id ) {
      streamDataExtractor.accumulatedData.id = chunk.id;
    }
    streamDataExtractor.accumulatedData.model = chunk.model;
    streamDataExtractor.accumulatedData.created = chunk.created;

    const choice = chunk.choices[0];
    if (choice.finish_reason) {
      streamDataExtractor.accumulatedData.choices[0].finish_reason = choice.finish_reason;
    }

    if (this.dialectResolver) {
      await this.dialectResolver.extractFromDelta('reasoning', choice.delta, streamDataExtractor as never);
    }

    await this.applyTextDelta(choice.delta, streamDataExtractor);
    await this.applyRefusalDelta(choice.delta, streamDataExtractor);
    await this.applyToolCallDelta(choice.delta, streamDataExtractor);
  }

  private async applyTextDelta(
    delta: ChatCompletionChunk.Choice.Delta,
    streamDataExtractor: StreamDataExtractor<ChatCompletion>,
  ): Promise<void> {
    if (!delta.content || delta.content.length === 0) {
      return;
    }

    await streamDataExtractor.accumulate(
      {
        key: 'content',
        value: delta.content,
      },
      (accumulated, newData) => {
        accumulated.choices[0].message.content = newData;
      },
      (_existing, newData, accumulated) => {
        accumulated.choices[0].message.content += newData;
      },
      (newData) => {
        if (!newData || newData.length === 0) {
          return null;
        }
        return { type: 'text', text: newData };
      },
    );
  }

  private async applyRefusalDelta(
    delta: ChatCompletionChunk.Choice.Delta,
    streamDataExtractor: StreamDataExtractor<ChatCompletion>,
  ): Promise<void> {
    if (!delta.refusal || delta.refusal.length === 0) {
      return;
    }

    await streamDataExtractor.accumulate(
      {
        key: 'refusal',
        value: delta.refusal,
      },
      (accumulated, newData) => {
        accumulated.choices[0].message.refusal = newData;
      },
      (_existing, newData, accumulated) => {
        accumulated.choices[0].message.refusal += newData;
      },
      (newData) => {
        if (!newData || newData.length === 0) {
          return null;
        }
        return { type: 'refusal', reason: newData };
      },
    );
  }

  private async applyToolCallDelta(
    delta: ChatCompletionChunk.Choice.Delta,
    streamDataExtractor: StreamDataExtractor<ChatCompletion>,
  ): Promise<void> {
    if (!delta.tool_calls) {
      return;
    }

    for (const tc of delta.tool_calls) {
      await streamDataExtractor.accumulate(
        {
          key: `tool_call_${tc.index}`,
          value: {
            type: 'function',
            id: tc.id ?? '',
            function: tc.function ?? { name: '', arguments: '' },
          } as ChatCompletionMessageFunctionToolCall,
        },
        (accumulated, newData) => {
          (accumulated.choices[0] as { message: { tool_calls: unknown[] | undefined } }).message.tool_calls ??= [];
          (accumulated.choices[0] as { message: { tool_calls: unknown[] } }).message.tool_calls.push(newData);
        },
        (existing, newData) => {
          const existingCall = existing as { id?: string; function?: { name?: string; arguments?: string } };
          const newCall = newData as { id?: string; function?: { name?: string; arguments?: string } };
          existingCall.id = existingCall.id || newCall.id || '';
          existingCall.function = existingCall.function ?? { name: '', arguments: '' };
          existingCall.function.name = existingCall.function.name || newCall.function?.name || '';
          existingCall.function.arguments += newCall.function?.arguments ?? '';
        },
        (newData) => ({
          type: 'tool_call',
          id: newData.id ?? '',
          name: newData.function?.name ?? '',
          arguments: newData.function?.arguments ?? '',
        }),
      );
    }
  }
}
