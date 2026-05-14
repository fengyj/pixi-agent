import { OpenAI } from 'openai';
import type { ChatCompletionFunctionTool, ChatCompletionMessageParam } from 'openai/resources/chat/completions/completions';
import { XMLBuilder } from 'fast-xml-parser';
import { z } from 'zod';
import { fakeToolset } from './tools';

const userTextMessageSchema = z.object({
    content: z.string().min(1),
    metadata: z.record(z.string(), z.string().or(z.number())).optional(),
})
type userTextMesssageType = z.infer<typeof userTextMessageSchema>;

const genUserTextMessage = (data: userTextMesssageType): string => {
  const validatedData = userTextMessageSchema.parse(data);

    const builder = new XMLBuilder({
        ignoreAttributes: false,
        format: false,
        suppressEmptyNode: true,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const xmlData: any = {
        message: {
            content: {
          '#cdata': validatedData.content,
            },
        },
    };

    const metadataEntries = Object.entries(validatedData.metadata ?? {});
    if (metadataEntries.length > 0) {
        xmlData.message.metadata = metadataEntries.map(([key, value]) => ({
            '@_name': key,
            '#text': String(value),
        }));
    }

    return builder.build(xmlData);
};

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
    content: `You are a helpful assistant. Always answer questions concisely and accurately.
  For the user message, it is an xml and the schema is as below:
<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"
           elementFormDefault="qualified">

  <xs:element name="message">
    <xs:complexType>
      <xs:sequence>
        <xs:element name="content" type="xs:string"/>
        <xs:element name="metadata"
                    type="metadataType"
                    minOccurs="0"
                    maxOccurs="unbounded"/>
      </xs:sequence>
    </xs:complexType>
  </xs:element>

  <xs:complexType name="metadataType">
    <xs:simpleContent>
      <xs:extension base="xs:string">
        <xs:attribute name="name" type="xs:string" use="required"/>
      </xs:extension>
    </xs:simpleContent>
  </xs:complexType>

</xs:schema>
The actual user message is in the content tag. The data of metadata tags are reference information.
You MUSTN'T use this format to response.
  `,
  },
  {
    role: 'user',
    content: genUserTextMessage({
      content: 'My name is Eric.', 
      metadata: {
        time: new Date().toISOString(),
        msg_id: '123456',
      }}),
  }
] as Array<ChatCompletionMessageParam>;

let response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
  prompt_cache_retention: 'in_memory',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);

messages.push(response.choices[0].message);
messages.push({
    role: 'user',
    content: genUserTextMessage({
      content: 'Do you know what time it is now? And get the price of the IBM today.'
    }),
})
/**
 * ```json
 * {
 *   "id":"chatcmpl-DcNZpPTnz90zao6VMJNWiruhbJcvB",
 *   "object":"chat.completion",
 *   "created":1778039465,
 *   "model":"openai/gpt-5.4-nano",
 *   "choices":[{"index":0,"message":{
 *         "role":"assistant",
 *         "tool_calls":[{
 *           "id":"call_MxA94ig0YozfmX3CGki1YTHX",
 *           "type":"function",
 *           "function":{"name":"stock_ohlc","arguments":"{\\"date\\":\\"2026-05-06\\",\\"ticker\\":\\"IBM\\"}"}
 *         }],
 *         "content":null},
 *       "finish_reason":"tool_calls"}],
 *   "usage":{"prompt_tokens":559,"completion_tokens":29,"total_tokens":588}}
 * ```
 */
response = await openai.chat.completions.create({
  model: model,
  messages: messages,
  tools: tools,
  max_tokens: 500,
  prompt_cache_retention: 'in_memory',
});
process.stdout.write(`Response:\n{${response.choices[0].message.content}}\n`);
messages.push(response.choices[0].message);