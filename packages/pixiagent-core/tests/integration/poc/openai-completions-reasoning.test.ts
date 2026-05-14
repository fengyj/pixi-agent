import { OpenAI } from 'openai';
import { ChatCompletionFunctionTool, ChatCompletionMessageParam } from 'openai/resources/chat/completions/completions';
import { executeToolCall, fakeToolset } from './tools';

const openai = new OpenAI({
  baseURL: 'https://api.ofox.ai/v1',
  apiKey: process.env.OFOXAI_API_KEY,
});

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

messages.push({
  role: 'user',
  content: 'Can you help to compare OFoxAI and OpenRouter? First think about what the advantages and disadvantages of each other are. Then summarize the comparison. Finally, give me the answer.',
})

/**
 * ```json
 * {
 *   "id":"chatcmpl-DcOqIJOBoguQBYEwIOg20dimVBl5q",
 *   "object":"chat.completion",
 *   "created":1778044330,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{
 *       "index":0,
 *       "message":{
 *         "role":"assistant",
 *         "content":"I can compare them at a high level, but I don’t have live product/price/feature data right now—so treat this as a conceptual comparison. If you share your priorities (cost, model quality, ease of use, region/data handling), I can tailor it.\\n\\n### OFoxAI vs OpenRouter (conceptual comparison)\\n\\n| Aspect | OFoxAI | OpenRouter |\\n|---|---|---|\\n| What it is | A platform/service that provides access to models (exact positioning varies by provider details). | A “model router”/marketplace that connects you to many different LLM providers through one API. |\\n| Model variety | Usually more limited than “router” platforms (often focuses on selected providers/models). | Typically very broad: many providers/models accessible via one endpoint. |\\n| Ease of switching models | Often straightforward within their catalog, but switching away from their supported set may be harder. | Designed for switching/routing across many models/providers. |\\n| Reliability / latency (practical) | Can be stable if they run/route in a controlled way; performance depends on their infrastructure and chosen provider. | Can vary by routed provider; latency/availability depends on which backend the router selects. |\\n| Pricing transparency | Depends on their specific pricing page; may be simpler if they aggregate pricing under one scheme. | Can be variable by model/provider; often competitive, but you must compare per-model rates. |\\n| Control over backend | Typically less granular (you’re selecting from their offerings). | Often more control/visibility because you choose specific models/providers (depending on setup). |\\n| Quotas / rate limits | Depends on their plan; usually tied to OFoxAI’s own quota policies. | Depends on the provider/model chosen and router rules; limits can differ by backend. |\\n| Data/privacy posture | Depends on their policy and how they handle prompts/logs; read their terms carefully. | Similar—also depends on underlying provider(s) because prompts may be sent to different backends. |\\n| Best fit | If you want a simpler “one platform” experience with fewer moving parts. | If you want flexibility: try many models, optimize cost/latency per model, and switch quickly. |\\n| Main risk/tradeoff | Less choice; you may hit limitations if a needed model/provider isn’t available. | More complexity; behavior/cost/performance can change across providers and model selections. |\\n\\n### Advantages / disadvantages of each"
 *       },
 *       "finish_reason":"length"
 *   }],
 *   "usage":{"prompt_tokens":435,"completion_tokens":500,"total_tokens":935}}
 * ```
 */
let response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  max_completion_tokens: 500,
  reasoning_effort: 'medium',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);

messages.push({
    role: 'user',
    content: 'Can you continue to finish the answer?',
})
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  max_completion_tokens: 3000,
  reasoning_effort: 'medium',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);
