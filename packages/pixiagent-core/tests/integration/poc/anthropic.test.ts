import Anthropic from '@anthropic-ai/sdk';
import { MessageParam, Tool } from '@anthropic-ai/sdk/resources';
import { executeToolCall, fakeToolset } from './tools';

const client = new Anthropic({
  baseURL: "https://api.ofox.ai/anthropic",
  apiKey: process.env.OFOXAI_API_KEY
});

const tools = fakeToolset.tools.map((tool) => ({
    input_schema: tool.definition.parameters,
    name: tool.definition.name,
    description: tool.definition.description,
  }) as Tool);

const model = 'anthropic/claude-haiku-4.5';

const instructions = 'You are a helpful assistant. Always answer questions concisely and accurately.'

const messages = [{
    role: 'user',
    content: 'Can you help to compare OFoxAI and OpenRouter? First think about what the advantages and disadvantages of each other are. Then summarize the comparison. Finally, give me the answer.',
}] as Array<MessageParam>;

/**
 * ```json
 * {
 *   "id":"msg_bdrk_013ZZx47LCW8NgpDfaN8g9Yc",
 *   "type":"message",
 *   "role":"assistant",
 *   "content":[{
 *       "type":"thinking",
 *       "thinking":"The user is asking me to compare \\"OFoxAI\\" and \\"OpenRouter\\". They want me to:\\n1. Think about advantages and disadvantages of each\\n2. Summarize the comparison\\n3. Give them the answer\\n\\nHowever, I need to consider what I know about these services. Let me think about this:\\n\\n- \\"OpenRouter\\" is a real service/platform that aggregates access to various AI models from different providers (like OpenAI, Anthropic, etc.), allowing users to access multiple models through a single API.\\n\\n- \\"OFoxAI\\" - I'm not immediately familiar with this as a well-known AI service or platform. It's possible this is:\\n  1. A lesser-known or newer service\\n  2. A misspelling\\n  3. A specific product I'm not familiar with\\n  4. A hypothetical service\\n\\nGiven that I don't have reliable information about \\"OFoxAI\\" and cannot access real-time information or specific details about services beyond my training data, I should be honest about this limitation. I should not make up information about a service I'm not certain about.\\n\\nThe user hasn't asked me to use any of the available tools (weather or stock data), and those tools wouldn't be relevant anyway for comparing AI services.\\n\\nI should ask the user for clarification about what \\"OFoxAI\\" is, or let them know that I'm not familiar with this service and ask for more information so I can provide an accurate comparison.",
 *       "signature":"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"
 *     },{
 *       "type":"text",
 *       "text":"I'd be happy to help you compare these services, but I need to let you know about a limitation:\\n\\nI'm **not familiar with \\"OFoxAI\\"** as a known AI service or platform. While I'm well-acquainted with **OpenRouter** (a platform that aggregates access to multiple AI models), I cannot find reliable information about OFoxAI.\\n\\n**Could you clarify:**\\n\\n1. **What is OFoxAI?** Is it:\\n   - A specific AI model provider or API service?\\n   - A newer platform I might not be aware of?\\n   - Possibly spelled differently?\\n   - A tool or service from a particular company?\\n\\n2. **What context are you comparing them in?**\\n   - API/model access?\\n   - Pricing models?\\n   - Specific features?\\n   - Use cases?\\n\\nOnce you provide more details about OFoxAI, I'll be able to give you a thorough and accurate comparison with OpenRouter, including advantages, disadvantages, and a clear summary."
 *     }],
 *   "model":"anthropic/claude-haiku-4.5",
 *   "usage":{
 *     "input_tokens":1157,
 *     "cache_creation_input_tokens":0,
 *     "cache_read_input_tokens":0,
 *     "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},
 *     "output_tokens":565
 *   },
 *   "stop_reason":"end_turn",
 *   "stop_sequence":null,
 *   "parsed_output":null
 * }
 * ```
 */
const response = await client.messages.stream({
  max_tokens: 1024,
  messages: messages,
  model: 'anthropic/claude-haiku-4.5',
  system: instructions,
  stream: true,
  thinking: {
    type: 'adaptive',
    display: 'summarized'
  },
  tools: tools,
}).on('text', chunk => {
    process.stdout.write(chunk);
}).on('thinking', chunk => {
    process.stdout.write(chunk);
}).on('error', error => {
    process.stderr.write(`Error: ${error.message}\n`);
}).on('abort', error => {
    process.stderr.write(`Aborted: ${error.message}\n`);
}).on('finalMessage', msg => {
    process.stdout.write('\n');
    messages.push(msg);
}).finalMessage();

process.stdout.write(`Usage: ${JSON.stringify(response.usage, null, 2)}\n`);

