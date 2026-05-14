import { OpenAI } from 'openai';
import { ChatCompletionFunctionTool, ChatCompletionMessageParam } from 'openai/resources/chat/completions/completions';
import { executeToolCall, fakeToolset } from './tools';
import { on } from 'events';

const openai = new OpenAI({
  baseURL: 'https://api.ofox.ai/v1',
  apiKey: process.env.OFOXAI_API_KEY,
});

const tools = fakeToolset.tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.definition.name,
      description: tool.definition.description,
      parameters: tool.definition.parameters,
    },
  }) as ChatCompletionFunctionTool);

const model = 'openai/gpt-5.4-nano';

const messages = [
  {
    role: 'developer',
    content: 'You are a helpful assistant. Always answer questions concisely and accurately.',
  },
  {
    role: 'user',
    content: 'My name is Eric.',
  }
] as Array<ChatCompletionMessageParam>;

/**
 * ```json
 * {
 *   "object":"chat.completion",
 *   "service_tier":"default",
 *   "usage":{
 *     "prompt_tokens":225,
 *     "prompt_tokens_details":{"cached_tokens":0},
 *     "completion_tokens":18,
 *     "completion_tokens_details":{},
 *     "total_tokens":243
 *   },
 *   "id":"chatcmpl-DcRa9bdJkBCBOPXLqevEARDrlEE3i",
 *   "choices":[{"message":{
 *       "role":"assistant",
 *       "content":"Nice to meet you, Eric! How can I help you today?",
 *       "refusal":null,
 *       "parsed":null
 *     },
 *     "finish_reason":"stop",
 *     "index":0,
 *     "logprobs":null
 *   }],
 *   "created":0,
 *   "model":""
 * }
 * ```
 */
let response = await openai.chat.completions.stream({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
  prompt_cache_retention: 'in_memory',
}).on('chunk', chunk=>{
    const delta = chunk.choices[0]?.delta as any
    if (delta?.reasoning_content)
        process.stdout.write(delta.reasoning_content);
}).on('refusal.delta', chunk => {
    process.stderr.write(chunk.delta);
}).on('refusal.done', chunk => {
    process.stderr.write('\n');
}).on('content.delta', chunk => {
    process.stdout.write(chunk.delta);
}).on('content.done', chunk => {
    process.stderr.write('\n');
}).on('functionToolCall', toolCall => {
    process.stdout.write(`tool call: ${toolCall.name}, args: ${toolCall.arguments}\n`);
}).on('finalMessage', message => {    
    messages.push(message);
}).finalChatCompletion();

messages.push({
    role: 'user',
    content: 'Do you know the weather tomorrow?',
})

/**
 * ```json
 * {
 *   "object":"chat.completion",
 *   "service_tier":"default",
 *   "usage":{
 *     "prompt_tokens":256,
 *     "prompt_tokens_details":{"cached_tokens":0},
 *     "completion_tokens":14,
 *     "completion_tokens_details":{},
 *     "total_tokens":270
 *   },
 *   "id":"chatcmpl-DcRdklyzO2RzbiJRmXG1p86Y4ZuTD",
 *   "choices":[{
 *       "index":0,
 *       "finish_reason":"tool_calls",
 *       "logprobs":null,
 *       "message":{
 *         "role":"assistant",
 *         "content":null,
 *         "refusal":null,
 *         "tool_calls":[{
 *           "id":"call_D0oqtndtoPit7Z1zTQqFqLBr",
 *           "type":"function",
 *           "function":{"name":"future_weather","arguments":"{}"}
 *         }],
 *         "parsed":null
 *       }
 *   }],
 *   "created":0,
 *   "model":""}
 * ```
 */
response = await openai.chat.completions.stream({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
}).on('chunk', chunk=>{
    const delta = chunk.choices[0]?.delta as any
    if (delta?.reasoning_content)
        process.stdout.write(delta.reasoning_content);
}).on('refusal.delta', chunk => {
    process.stderr.write(chunk.delta);
}).on('refusal.done', chunk => {
    process.stderr.write('\n');
}).on('content.delta', chunk => {
    process.stdout.write(chunk.delta);
}).on('content.done', chunk => {
    process.stderr.write('\n');
}).on('functionToolCall', toolCall => {
    process.stdout.write(`tool call: ${toolCall.name}, args: ${toolCall.arguments}\n`);
}).on('finalMessage', message => {    
    messages.push(message);
}).finalChatCompletion();

if (response.choices[0].message.tool_calls) {
    for (const call of response.choices[0].message.tool_calls) {
        if (call.type !== 'function') continue;

        const toolCallResult = await executeToolCall(
            call.function.name, 
            JSON.parse(call.function.arguments), 
            fakeToolset.tools,
        );
        process.stdout.write(`Tool call result:\n${JSON.stringify(toolCallResult)}\n`);
        messages.push({
            role: 'tool',
            tool_call_id: call.id,
            content: JSON.stringify(toolCallResult),
        });
    }
}
else {
    throw new Error('No tool call found in the response');
}

/**
 * ```json
 * {
 *   "object":"chat.completion",
 *   "service_tier":"default",
 *   "usage":{"prompt_tokens":352,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens":40,"completion_tokens_details":{},"total_tokens":392},
 *   "id":"chatcmpl-DcRlxF9pn53vihWFrWUZuBQ459sKe",
 *   "choices":[{
 *       "message":{
 *         "role":"assistant",
 *         "content":"Tomorrow (2026-05-07) is expected to be **rainy**, with a **high of 21°C** and a **low of 12°C**.",
 *         "refusal":null,
 *         "parsed":null
 *       },
 *       "finish_reason":"stop",
 *       "index":0,"logprobs":null
 *   }],
 *   "created":0,
 *   "model":""}
 * ```
 */
response = await openai.chat.completions.stream({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
}).on('chunk', chunk=>{
    const delta = chunk.choices[0]?.delta as any
    if (delta?.reasoning_content)
        process.stdout.write(delta.reasoning_content);
}).on('refusal.delta', chunk => {
    process.stderr.write(chunk.delta);
}).on('refusal.done', chunk => {
    process.stderr.write('\n');
}).on('content.delta', chunk => {
    process.stdout.write(chunk.delta);
}).on('content.done', chunk => {
    process.stdout.write('\n');
}).on('functionToolCall', toolCall => {
    process.stdout.write(`tool call: ${toolCall.name}, args: ${toolCall.arguments}\n`);
}).on('finalMessage', message => {    
    messages.push(message);
}).finalChatCompletion();

process.stdout.write('\n');