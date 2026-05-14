import { OpenAI } from 'openai';
import { ChatCompletionFunctionTool, ChatCompletionMessageParam } from 'openai/resources/chat/completions/completions';
import { executeToolCall, fakeToolset } from './tools';

const openai = new OpenAI({
  baseURL: 'https://api.ofox.ai/v1',
  apiKey: process.env.OFOXAI_API_KEY,
});

const tools = fakeToolset.tools.map((tool) => ({
    type: 'function',
    function: tool.definition,
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
 *   "id":"chatcmpl-DcLqrdcSpeqTUYnTB0kmRNP5XxXG0",
 *   "object":"chat.completion",
 *   "created":1778032833,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{
 *       "role":"assistant",
 *       "content":"Hi Eric! How can I help you today?"},
 *     "finish_reason":"stop"}],
 *   "usage":{"prompt_tokens":225,"completion_tokens":14,"total_tokens":239}
 * }
 * ```
 */
let response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);

messages.push(response.choices[0].message);
messages.push({
    role: 'user',
    content: 'Do you know the weather tomorrow?',
})

/**
 * ```json
 * {
 *   "id":"chatcmpl-DcMHzXHhNQZDwzEda9fKYTqU8VuMX",
 *   "object":"chat.completion",
 *   "created":1778034515,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{
 *       "role":"assistant",
 *       "tool_calls":[{
 *         "id":"call_SLc8n89o1vuumYQAeJemVuZW",
 *         "type":"function",
 *         "function":{
 *           "name":"future_weather",
 *           "arguments":"{}"}}],
 *       "content":null},
 *     "finish_reason":"tool_calls"}],
 *   "usage":{"prompt_tokens":256,"completion_tokens":14,"total_tokens":270}
 * }
 * ```
 */
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  tools: tools,
  max_completion_tokens: 500,
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);

if (response.choices[0].message.tool_calls) {
    messages.push(response.choices[0].message);
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
 *   "id":"chatcmpl-DcMllulYr3wSY7XukpjDbkpJb8FdH",
 *   "object":"chat.completion",
 *   "created":1778036361,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{
 *         "role":"assistant",
 *         "content":"Yes—tomorrow (2026-05-07) it’s forecast to be **21°C / 12°C** with **Rainy** weather."},
 *       "finish_reason":"stop"}],
 *   "usage":{"prompt_tokens":352,"completion_tokens":36,"total_tokens":388}
 * }
 * ```
 */
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);

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
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  max_completion_tokens: 500,
  reasoning_effort: 'medium',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);

// OpenAI's models don't return thinking info via the completion API.
messages.push({
    role: 'user',
    content: 'Can you continue to finish the answer?',
})
/**
 * ```json
 * {
 *   "id":"chatcmpl-DcQLhAV0udyx35w0WTd5IprjFXXsd",
 *   "object":"chat.completion",
 *   "created":1778050121,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{"role":"assistant","content":"Sure—I’ll finish, but quick note: I don’t have reliable, up-to-date public details about **OFoxAI** specifically (features, pricing, supported models, uptime guarantees). So I’ll compare them in the most useful way based on what these services typically are: **model/API gateways** that route requests to LLM providers.\\n\\n## Advantages & disadvantages (continued)\\n\\n### **OFoxAI** *(assumed: an LLM/API gateway similar to OpenRouter)*\\n**Advantages**\\n- **Potentially simpler setup** than highly “routing-heavy” platforms (depends on how they expose models).\\n- **May offer better value** (pricing/promos/tiers) if they have favorable upstream agreements.\\n- **May have more consistent behavior** if they use fewer upstreams or fixed model mappings.\\n- **Could be region-optimized** (latency) depending on where they host/relay traffic.\\n\\n**Disadvantages**\\n- **Model availability may be narrower** than OpenRouter’s (depends on their catalog).\\n- **Less transparency/control** if they route through fewer providers or don’t expose routing details.\\n- **Smaller ecosystem/support/community** relative to OpenRouter (affects troubleshooting/examples).\\n- **Feature parity risk**: depending on implementation, function calling, tool support, streaming behavior, JSON reliability, etc. may differ.\\n\\n### **OpenRouter** *(from before, summarized fully)*\\n**Advantages**\\n- **Largest breadth of models** and active experimentation.\\n- **Good for “pick the best model” workflows** (you can test across providers).\\n- **Usually OpenAI-compatible**, making integration easier.\\n- **Competitive pricing** due to multiple upstreams.\\n\\n**Disadvantages**\\n- **Variability**: quality/latency can fluctuate because routing depends on which upstream is chosen.\\n- **More variables to debug** (model/provider combinations, rate limits, routing decisions).\\n- **Sometimes “best-effort” behavior**: performance may not be as consistent as a single-provider setup.\\n\\n---\\n\\n## Summary comparison\\n- Choose **OpenRouter** if you want **the widest model selection**, strong experimentation, and potentially better pricing—but accept **some variability** in quality/latency.\\n- Choose **OFoxAI** if it offers **better consistency, simpler integration, or better pricing** for the models you care about—but model variety and ecosystem support may be more limited (and we’d need to verify).\\n\\n---\\n\\n## My answer (recommendation)\\nIf your top priority is **maximum model choice and flexibility**, go with **OpenRouter**.  \\nIf your top priority is **consistency / simpler workflow / better cost for a specific subset of models**, then **OFoxAI** could be better—**but you should confirm**: supported models, pricing, streaming/function-calling behavior, and whether routing/latency is consistent.\\n\\nIf you tell me what you care about most (cost, latency, specific models, tool/function calling, reliability, region/compliance), I can give a sharper “pick A vs B” recommendation."},"finish_reason":"stop"}],
 *   "usage":{
 *     "prompt_tokens":374,
 *     "completion_tokens":834,
 *     "total_tokens":1208,
 *     "completion_tokens_details":{"reasoning_tokens":210}
 *   }
 * }
 * ```
 */
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  max_completion_tokens: 3000,
  reasoning_effort: 'medium',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);

messages.push({
        role: "user",
        content: [
          { type: "text", text: "What can you see in this image?" },
          {
            type: "image_url",
            image_url: {
              "url": "https://n.sinaimg.cn/sinacn09/559/w870h489/20180721/8468-hfqtahi0947837.jpg",
            },
          }
        ],
      });
/**
 * ```json
 * {
 *   "id":"chatcmpl-DcQPmvgH4WLQPLUFXIzrpC9U8uM8C",
 *   "object":"chat.completion",
 *   "created":1778050374,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{
 *         "role":"assistant",
 *         "content":"The image shows a dramatic snow-covered mountain peak with rocky faces and white glaciers. On the summit there’s a small building/structure with a tall spire (tower) and what looks like mountain infrastructure (possibly a station/antenna). The foreground includes snowy ridges and rails/paths, with distant mountains fading into a cloudy valley under a bright blue sky."},"finish_reason":"stop"
 *   }],
 *   "usage":{
 *     "prompt_tokens":1531,
 *     "completion_tokens":193,
 *     "total_tokens":1724,
 *     "completion_tokens_details":{"reasoning_tokens":110}
 *   }
 * }
 * ```
 */
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  max_completion_tokens: 500,
  reasoning_effort: 'medium',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);
