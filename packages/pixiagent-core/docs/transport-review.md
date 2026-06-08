# Review: `response.ts` & `chat_completion.ts` — Type Conversion Issues

## 1. `response.ts` — Issues

### 1.1 (Critical) `fromResponseReasoningItem` drops reasoning content — only keeps summary

**Location:** `ResponseContentPartHelper.fromResponseReasoningItem()` (~line 560)

```ts
fromResponseReasoningItem(item: ResponseReasoningItem): ThinkingPart | null {
  const summary = Array.isArray(item.summary)
    ? item.summary.map((summaryItem) => summaryItem.text).join('')
    : '';
  if (!summary) {
    return null;
  }
  return { type: 'thinking', content: summary };
}
```

My questions:
1. the text in the content is encrypted or not? if it's not encrypted, when will model exports summary and when export content? if both have value, which one should be exposed to the user? and as the input in next turn if it's used in another api mode?
2. if i get the reasoning text from another model via another kind of API mode, should i put the text into summary or content?

### Answers to 1.1 questions

**Q1: Is `content` encrypted? When does the model emit `summary` vs `content`?**

`ResponseReasoningItem.content` is **not** encrypted — it's the actual human-readable chain-of-thought text (`reasoning_text` blocks). Encryption is indicated by a **separate** field `encrypted_content` (a base64 string). When `encrypted_content` is present, the `content` array is typically empty or absent. So these are mutually exclusive:

| Scenario | `content` | `summary` | `encrypted_content` |
|---|---|---|---|
| Normal reasoning (visible) | ✅ present | may be present | absent |
| Encrypted reasoning (hidden) | absent/empty | may be present | ✅ present |

Per OpenAI's API behavior:
- `summary` is always safe to expose — it's a sanitized summary designed for end-user display.
- `content` is the raw thinking — also human-readable, but providers reserve the right to encrypt it in future. When encrypted, `content` becomes empty and `encrypted_content` takes over.
- The model may emit **both** `content` and `summary` (common: detailed reasoning in content, brief summary in summary). If both exist, `content` is the full reasoning trace while `summary` is a condensed version.

**Which one should be exposed to the user?**
- `summary` is always safe and designed for user display.
- `content` is also safe **as long as `encrypted_content` is absent**. If you want to expose the full reasoning, use `content`; if you want a concise version, use `summary`.
- **Recommendation**: expose `content` as the primary reasoning text, and if it's empty, fall back to `summary`. Always check `encrypted_content` first — if present, only use `summary` (see issue 1.2).

**As input for next turn in another API mode?**
Only `summary` should be used as input for the next turn. Raw `content` (detailed reasoning) is **not** intended to be fed back to the model — providers like Anthropic and OpenAI explicitly state that reasoning tokens should not be included in subsequent requests. `summary` on the other hand is designed to be safe for re-injection. When converting to other API modes:
- For Anthropic: put summary text into a `ThinkingBlock` (or just skip it — Anthropic doesn't need prior reasoning as input).
- For Chat Completions: reasoning should generally NOT be included in the message array at all — it's output-only metadata. The dialect resolvers (DeepSeek, OpenRouter) handle this by stripping reasoning when building the request params.

**Q2: If I get reasoning text from another API mode, should it go into `summary` or `content`?**

When converting from another API mode (e.g., Anthropic thinking blocks, DeepSeek `reasoning_content`) into a `ResponseReasoningItem`:

- Put it into **`summary`** (as `summary_text` blocks). This is the safer choice because:
  1. `summary` is the cross-provider compatible field — all Response-compatible APIs support it.
  2. `content` (`reasoning_text`) is more provider-specific and may be encrypted by some providers.
  3. When this Response message is later fed back to the model, only `summary` will be preserved (reasoning content is typically stripped before the next request).

- Only put into `content` if you're certain the source data is unencrypted, human-readable reasoning AND you want to preserve the full detail. But even then, it won't survive a round-trip as input.

**Practical recommendation**: Always use `summary` for cross-API-mode reasoning data. Reserve `content` for cases where you're faithfully reproducing an original Response API reasoning item.

`ResponseReasoningItem` has TWO data fields:
- `content`: the actual chain-of-thought reasoning text (`Array<{type:'reasoning_text', text:string}>`)
- `summary`: a summarized version

This function **only reads `summary`** and completely ignores `content`. So during `convertFromRawMessage`, the *actual reasoning* (the step-by-step thinking) is silently discarded. Only the summary survives into the `SessionMessage`.

**Impact:**
- When switching API modes (e.g., from Response to Anthropic), the detailed reasoning is lost.
- For the same-API-mode next turn this is fine (raw message is used directly), but cross-mode conversion is degraded.

**Example of what's lost:**

```
Raw ResponseReasoningItem:
{
  type: "reasoning",
  content: [
    { type: "reasoning_text", text: "Let me think... first we need to..." },
    { type: "reasoning_text", text: "Then consider the edge case..." }
  ],
  summary: [
    { type: "summary_text", text: "Considered edge cases and chose approach A." }
  ]
}

→ Converted to ThinkingPart:
{ type: "thinking", content: "Considered edge cases and chose approach A." }
// ↑ "Let me think... first we need to..." is GONE
```

**In streaming (`generate`),** the reasoning text deltas ARE emitted as ThinkingPart chunks (via `response.reasoning_text.delta` → `{ type: 'thinking', content: delta.text }`). But the final `convertFromRawMessage` still drops them. So the streaming callbacks see the reasoning, but the persisted session message does not.

**Fix:** `fromResponseReasoningItem` should also process `item.content` and produce thinking parts from it.

---

### 1.2 (Critical) No check for encrypted reasoning content

**Location:** Multiple places in `generate()` and `fromResponseReasoningItem`

`ResponseReasoningItem` can have an `encrypted_content` field (a base64 string). When present, the `content` array may be empty or absent — the real reasoning is encrypted and only decipherable by the original provider.

The code never checks for `encrypted_content`. If a provider sends:
```json
{
  "type": "reasoning",
  "encrypted_content": "AES-256-GCM...",
  "summary": [{ "type": "summary_text", "text": "..." }]
}
```

The summary would be converted to a `ThinkingPart` (which is fine — summaries are designed to be safe), but there's no guard against accidentally processing `content` alongside `encrypted_content`.

More critically, in the streaming path (`response.reasoning_text.delta`), if a third-party Response-compatible API emits reasoning deltas that are encrypted but not marked, they would be emitted as plain `{ type: 'thinking', content: ... }` chunks — leaking garbled/encrypted text to the UI.

**What other dialects do:**
- **OpenRouter dialect** explicitly checks for `reasoning.encrypted` and **ignores** it.
- **Anthropic transport** handles `RedactedThinkingBlock` by skipping it.

The Response transport should similarly inspect `encrypted_content` and suppress content conversion when it's present.

---

My questions:
1. how can i know what reasoning data is safe to expose to users?

### Answers to 1.2 questions

**Q1: How can I know what reasoning data is safe to expose to users?**

Each provider/API mode has different signals. Here's a summary:

| Provider / API | Safe to expose | Unsafe / Must hide | How to detect |
|---|---|---|---|
| **Anthropic Messages** | `ThinkingBlock` (has `thinking` + `signature`) | `RedactedThinkingBlock` (has `data` field) | Check `block.type === 'redacted_thinking'` |
| **OpenAI Response** | `content` (reasoning_text) when `encrypted_content` is absent; `summary` is always safe | `encrypted_content` is present → `content` is empty | Check `item.encrypted_content` field |
| **OpenRouter (Chat)** | `reasoning.text`, `reasoning.summary` | `reasoning.encrypted` | Check `detail.type === 'reasoning.encrypted'` |
| **DeepSeek (Chat)** | `reasoning_content` field | Not applicable (DeepSeek doesn't encrypt reasoning) | Always safe |
| **Ofox** | standard delta fields | N/A (currently no encrypted reasoning) | N/A |

**General rule**: If the data comes through a field/block whose name or type explicitly indicates encryption (e.g., `encrypted_content`, `redacted_thinking`, `reasoning.encrypted`), do NOT convert it to `ThinkingPart`. If it comes through a regular text field (`reasoning_content`, `reasoning.text`, `reasoning_text`, `thinking`), it's safe.

**Practical implementation**: At the transport level, check for encryption markers before converting. At the `ThinkingPart` level, you could add a boolean field `encrypted: true` so UI layers can also make decisions (e.g., show a lock icon instead of garbled text). Currently, the code already correctly skips Anthropic's `RedactedThinkingBlock` and OpenRouter's `reasoning.encrypted` — but `response.ts` does NOT check `encrypted_content` (this is the gap to fix).

### 1.3 (High) `toItemFromServerToolUse` / `toItemFromServerToolUseForInput` — unsafe `JSON.parse`

**Location:** `ContentPartResponseHelper.toItemFromServerToolUse()` (~line 730) and `toItemFromServerToolUseForInput()`

```ts
toItemFromServerToolUse(part: ServerToolUsePart): ... {
  if (part.providerSpecific !== ApiModes.RESPONSE) {
    return { type: 'output_text', text: `Tool use: ${part.name} with data ${part.data}`, ... };
  } else {
    return {
      ...JSON.parse(part.data ?? '{}'),  // ← can throw
      type: part.name,
      call_id: part.id,
    };
  }
}
```

`part.data` comes from user/tool code that produces `ServerToolUsePart`. If the data string is malformed JSON, `JSON.parse` throws an unhandled exception. There's no try-catch here, unlike `toToolResult` which wraps its parse in try-catch.

**Fix:** Wrap in try-catch, fall back to treating as `output_text` / `input_text`.

---

### 1.4 (High) `fromResponseFunctionCallOutput` — lossy JSON round-trip for structured output

**Location:** `ResponseContentPartHelper.fromResponseFunctionCallOutput()` (~line 532)

```ts
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
        .filter((part) => part !== undefined),
    ),
  };
}
```

When `item.output` is `Array<ResponseInputText | ResponseInputImage | ResponseInputFile>`:
1. Each element is converted via `fromResponseObject()` → `ContentPart` (e.g., `TextPart`, `ImagePart`)
2. Filtered for non-null
3. `JSON.stringify`'d

The problem: `fromResponseObject` transforms the structure. For example:
- Input: `{ type: 'input_image', image_url: 'https://...', detail: 'auto' }`  
- After `fromResponseObject`: `{ type: 'image', image: { sourceType: 'url', url: 'https://...' } }`

So the JSON stored in `ToolResultPart.result` is a **ContentPart-format** object, not the original Response-format object. When converting back in `toToolResult`, the code checks `parsedResult.every(c => c.type === 'image' && 'image' in c ...)` — it expects ContentPart types, which matches. So the round-trip works **within** this codebase.

But if this `ToolResultPart.result` is consumed by external code expecting the original Response API format, it gets a different structure.

Additionally, `JSON.stringify` drops `undefined` values and doesn't preserve `Date` objects. If `fromResponseObject` ever produces a ContentPart with non-serializable fields, they are silently lost.

---

### 1.5 (High) `fromResponseSpecificToolCall` — no `call_id` path creates `ServerToolUsePart` with undefined `id`

**Location:** `ResponseContentPartHelper.fromResponseSpecificToolCall()` (~line 585)

```ts
fromResponseSpecificToolCall(item: ...): ToolCallPart | ServerToolUsePart {
  if ('call_id' in item && item.call_id) {
    const { call_id, type, ...rest } = item;
    return {
      type: 'tool_call',
      id: call_id,
      name: type,
      arguments: JSON.stringify(rest),
      providerSpecific: ApiModes.RESPONSE,
    };
  } else {
    const { type, ...rest } = item;
    return {
      type: 'server_tool_use',
      name: type,
      data: JSON.stringify(rest),
      providerSpecific: ApiModes.RESPONSE,
    };
  }
}
```

In the `else` branch (no `call_id`), the resulting `ServerToolUsePart` has no `id`. When converting back via `toItemFromServerToolUse`:

```ts
return {
  ...JSON.parse(part.data ?? '{}'),
  type: part.name,
  call_id: part.id,  // ← undefined!
};
```

The rebuilt object has `call_id: undefined`. This may violate the expected type shape for some Response API tool call types (like `ResponseComputerToolCall` which likely expects `call_id` to be a string).

---

### 1.6 (Medium) `toToolResult` — silent try-catch swallows structured data

**Location:** `ContentPartResponseHelper.toToolResult()` (~line 770)

```ts
try {
  const parsedResult = ... ? JSON.parse(part.result) : null;
  // complex logic to detect structured ContentPart arrays...
} catch {
  // empty — silently falls through to default
}
return {
  type: 'function_call_output',
  call_id: part.id,
  output: part.result ?? '',
  status: 'completed',
};
```

If `part.result` is almost-valid JSON (e.g., truncated), or if the complex detection logic throws (e.g., accessing properties on a primitive), the error is silently swallowed and the result is treated as a plain string. The tool output structure is lost.

**Recommendation:** At minimum, log a warning. Consider whether the fallback should include an `isError: true` flag.

---

### 1.7 (Medium) `convertFromRawMessage` — `EasyInputMessage` type guard is too loose

**Location:** `ResponseContentPartHelper.toContentPartsFromResponseItem()` (~line 475)

```ts
const isResponseMessageItem = (data): data is EasyInputMessage | ... => {
  return (
    data.type === 'message' ||
    (!data.type &&                    // ← no type field
      'content' in data &&
      (typeof data.content === 'string' ||
        (Array.isArray(data.content) && ...)))
  );
};
```

The `!data.type` check matches ANY object that lacks a `type` field but has a `content` field. This could match unexpected objects (e.g., error payloads, metadata objects) that happen to have a `content` property. While the downstream `fromResponseMessageItem` would likely handle them (by stringifying or mapping), it's a loose type guard.

My questions:
1. looks like i only need to check the type if it's `message`, right?

### Answers to 1.7 questions

**Q1: Do I only need to check `type === 'message'`?**

**No**, you also need the `!data.type` fallback. Here's why:

The `EasyInputMessage` type in OpenAI's SDK represents a simplified user message format where you omit the `type` field entirely:

```ts
// This is an EasyInputMessage — no type field:
{ role: 'user', content: 'hello' }

// vs. explicit message type:
{ type: 'message', role: 'user', content: 'hello' }
```

Both are valid `ResponseInputItem` variants that represent a user message. The `!data.type` check is specifically to catch `EasyInputMessage` shapes. If you remove it and only check `data.type === 'message'`, messages sent in the simplified format would not be recognized as messages and would fall through to `fromResponseObject()` which would return `null` for them (since they have no `type` field), causing the content to be silently dropped.

The concern I raised is that `!data.type && 'content' in data` is a loose check — any object without `type` but with `content` would match. However, in practice, within the `ResponseInputItem | ResponseOutputItem` union, the only items that can lack a `type` field are `EasyInputMessage` and `ItemReference` (which also lacks `type`). The additional `content` check filters out `ItemReference`. So the guard is actually precise enough for the current SDK types.

**Verdict**: Keep both checks. The `!data.type` path is necessary for `EasyInputMessage`. If you want to be stricter, you could add a `role` check: `!data.type && 'role' in data && 'content' in data`.

---

### 1.8 (Medium) `generate` — `response.reasoning_summary_text.delta` merge type mismatch

**Location:** `generate()` method, `response.reasoning_summary_text.delta` case (~line 310)

```ts
case 'response.reasoning_summary_text.delta': {
  streamDataExtractor.accumulate(
    {
      key: `output_item_${event.output_index}_summary_${event.summary_index}`,
      value: {
        type: 'summary_text',
        text: event.delta,
      },
    },
    // ...
    (existing, newData) => {
      existing.text += newData.text;
    },
    // ...
  );
}
```

The `key` is the same as in `response.reasoning_summary_part.added`, so the `existing` value comes from the previous `event.part` (a `ResponseReasoningItem.Summary` object). The `newData` value is `{ type: 'summary_text', text: event.delta }`. Both have a `text` property so the merge `existing.text += newData.text` works at runtime. But the types are technically mismatched — `existing` is typed as the `value` from the first accumulation and `newData` as the `value` from this one.

This is fragile: if the `ResponseReasoningItem.Summary` type ever changes (e.g., `text` becomes `content`), this silently breaks.

---

## 2. `chat_completion.ts` — Issues

### 2.1 (Resolved ✅) Custom/non-function tool calls lose type on round-trip

**Location:** `getFromAssistantMessageParam()` and `getAssistantMessageParam()`

**Update after code changes:** The user has fixed this. The two sides now work symmetrically:

**raw → session** (line ~110):
```ts
if (toolCall.type === 'function') {
  // standard function call — normal conversion
  parts.push({ type: 'tool_call', id: toolCall.id, name: toolCall.function.name,
               arguments: toolCall.function.arguments });
  return;
}
// For custom/non-function: store original type as name, rest as JSON
const { id, type, ...rest } = toolCall;
parts.push({
  type: 'tool_call', id, name: type,
  arguments: JSON.stringify(rest),
  providerSpecific: ApiModes.COMPLETIONS,
});
```

**session → raw** (line ~520):
```ts
if (part.providerSpecific === ApiModes.COMPLETIONS) {
  const parsedArguments = JSON.parse(part.arguments);
  if (parsedArguments && typeof parsedArguments === 'object') {
    return { ...parsedArguments, type: part.name, id: part.id };
  }
}
// fallback: standard function call
return { type: 'function', id: part.id, function: { name: part.name, arguments: part.arguments } };
```

This round-trips correctly. Example:
- Original: `{ type: "custom", id: "abc", custom_field: "val" }`
- In Session: `{ type: "tool_call", id: "abc", name: "custom", arguments: '{"custom_field":"val"}', providerSpecific: "completions" }`
- Back to raw: `{ type: "custom", id: "abc", custom_field: "val" }` ✅

**Remaining minor concerns:**
- Old persisted messages using the previous format (`ContentPart.createProviderToolCallArguments('openai_chat', ...)`) will NOT be recognized by the new session→raw code and will fall back to `type: 'function'`. If you have stored conversations, this is a migration concern.
- The `JSON.parse` in session→raw is wrapped in try-catch, so malformed `arguments` gracefully fall back to standard function call. Good.
- The spread `{ ...parsedArguments, type: part.name, id: part.id }` correctly overwrites any stale `type`/`id` that might have been in `rest`.

---

### 2.2 (Critical) No reasoning_content handling in `generate()` without dialect resolver

**Location:** `generate()` method (~line 430)

```ts
for await (const chunk of stream) {
  // ...
  if (this.dialectResolver) {
    await this.dialectResolver.extractFromDelta('reasoning', choice.delta, streamDataExtractor);
  }
  // ...
}
```

The `ChatCompletionTransport.generate()` method **only** delegates reasoning extraction to the dialect resolver. If there is NO dialect resolver configured:

- DeepSeek sends `reasoning_content` in the delta → **silently dropped**
- OpenRouter sends `reasoning_details` in the delta → **silently dropped**
- Any provider that puts thinking content in the delta → **lost**

This means that using the Chat Completions transport without a proper dialect resolver on a reasoning-capable provider results in complete loss of reasoning/thinking content.

**This is likely the case for many third-party OpenAI-compatible providers** that may add `reasoning_content` to their chat completion responses. The core transport should at minimum log a warning when it detects reasoning fields in deltas that it can't process.

---

### 2.3 (High) Legacy `function_call` produces `ToolCallPart` with empty `id`

**Location:** `getFromAssistantMessageParam()` (~line 95)

```ts
if (msg.function_call) {
  parts.push({
    type: 'tool_call',
    id: '',              // ← empty string
    name: msg.function_call.name,
    arguments: msg.function_call.arguments,
  } as ToolCallPart);
}
```

The deprecated `function_call` field (pre-`tool_calls` API) has no `id`. The code assigns an empty string. If this tool call later needs to be matched with a `tool_result` (which uses `tool_call_id`), there's no way to correlate them — all legacy function calls have `id: ''`.

This is low-risk since `function_call` is long-deprecated, but if it's encountered, the tool matching will be silently broken.

---

### 2.4 (Medium) Tool message split can produce orphaned user messages

**Location:** `getToolMessageParam()` (~line 410)

```ts
const rawMessages: ChatCompletionMessageParam[] = toolResults.map(...);

if (otherParts.length > 0) {
  rawMessages.push(
    this.getUserMessageParam({
      ...msg,
      role: 'user',
      content: otherParts,
    } as SessionMessage),
  );
}
```

When a tool message has both `tool_result` parts and other content (text, images, etc.), the other content becomes an extra **user** message appended after all the tool messages. This is the correct API behavior (Chat Completions requires `tool` messages to only contain tool results). But it changes the message sequence: the user message is injected *after* all tool results, which may not match the original intent.

For example, if the original message had: `[tool_result_A, text_explanation, tool_result_B]`, it becomes:
```
tool: result of A
tool: result of B
user: "text_explanation"
```

The `text_explanation` is now a user message *after* all tool results, effectively asking the model to respond to it as a new user input.

---

### 2.5 (Medium) No encrypted reasoning guard in core transport

**Location:** `generate()` method — absence of check

While the OpenRouter dialect correctly skips `reasoning.encrypted` blocks, the core `ChatCompletionTransport` itself has no such guard. If a provider emits encrypted reasoning through a non-standard delta field that the dialect resolver doesn't handle, encrypted text could leak into session messages.

This is mitigated by the fact that unknown delta fields are simply ignored (not processed). But if a future dialect resolver or the core transport starts handling a new delta field without checking for encryption, this becomes a risk.

---

### 2.6 (Low) `getFromToolMessageParam` simplification — array content now stringified

**Location:** `getFromToolMessageParam()` (~line 240)

**Change observed:** The old code split array tool content into individual `tool_result` parts (one per text block). The new code collapses everything:

```ts
// New code:
{
  type: 'tool_result',
  id: msg.tool_call_id,
  result: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
}
```

**Impact:** When `msg.content` is an array like `[{type:'text', text:'A'}, {type:'text', text:'B'}]`, the old code produced two `ToolResultPart` entries, each with a readable string result. The new code produces **one** `ToolResultPart` with `result: '[{"type":"text","text":"A"},{"type":"text","text":"B"}]'` — a JSON string of the raw array. This is less readable and the internal `type:'text'` wrappers are now visible in the result. However, the downstream `getToolMessageParam()` handles this correctly (it tries `JSON.parse` and detects the array-of-text pattern), so the round-trip still works. This is a low-severity cosmetic/structural change.

---

## 3. Cross-cutting Concerns

### 3.1 Reasoning/Thinking content — when is it safe to expose?

The `ThinkingPart` type has an optional `signature` field:

```ts
const ThinkingPartSchema = z.object({
  type: z.literal('thinking'),
  content: z.string(),
  signature: z.string().optional(),
});
```

This signature is used by Anthropic's thinking blocks to verify authenticity. But there's no field to indicate "this content is encrypted/pseudonymized and should NOT be shown to the user." The encryption status is only known at the transport/dialect level.

**What should happen with encrypted reasoning:**
- **Anthropic `RedactedThinkingBlock`**: has `data: string` (encrypted) — should NOT be converted to `ThinkingPart`. The Anthropic transport handles this correctly by skipping these blocks.
- **OpenRouter `reasoning.encrypted`**: explicitly ignored in the dialect.
- **OpenAI Response `encrypted_content`**: not handled in `response.ts`.
- **Chat Completions dialects**: rely on the dialect resolver to filter.

**Recommendation:** Add a field to `ThinkingPart` to mark encrypted/unreadable content, or add a separate `EncryptedThinkingPart` type. This would make the data flow explicit: transports set the flag, and UI layers can decide to hide/show accordingly.

My questions:
1. do you mean the data in the `RedactedThinkingBlock` should not be converted to the session message, because it's encrypted (it's useless for others, for the same provider, no needs to convert the raw message to session message).
2. can you tell me if there is any code wrong which converts the un-human-readable reasoning data to thinkingpart?

### Answers to 3.1 questions

**Q1: Should `RedactedThinkingBlock` data be converted to a SessionMessage?**

Yes, that's exactly what I mean — and it's actually what your Anthropic transport **already does correctly**. `RedactedThinkingBlock` contains encrypted data (`data: string`) that is only meaningful to Anthropic's servers. It should **not** be converted to `ThinkingPart`. Two reasons:

1. **It's useless for other providers**: The encrypted blob is Anthropic-specific and cannot be decrypted by any other API. Converting it to a `ThinkingPart` would mean persisting garbled base64 as "thinking content" — misleading and wasteful.
2. **Same provider doesn't need it in SessionMessage**: When the next turn uses the same Anthropic model, the raw `AnthropicApiMessage` is used directly (not converted through SessionMessage). So the encrypted block is preserved in `rawMessage` and doesn't need to be in the SessionMessage at all.

Check your `anthropic.ts` — it should already skip `RedactedThinkingBlock` during `convertFromRawMessage`. If it does, no changes needed.

**Q2: Is there any code that incorrectly converts un-human-readable reasoning to ThinkingPart?**

Let me trace through each code path:

**`anthropic.ts`** — ✅ Safe (assuming it skips `RedactedThinkingBlock`). Let me verify:
- `ThinkingBlock` (with `thinking` + `signature`): correctly converted to `ThinkingPart` with `signature` preserved.
- `RedactedThinkingBlock` (with `data`): should be skipped entirely.

**`chat_completion.ts` (OpenRouter dialect)** — ✅ Safe. The dialect explicitly checks:
```ts
// In extractFromDelta:
d.type === 'reasoning.text' || d.type === 'reasoning.summary'
// reasoning.encrypted is NOT in this list → ignored
```

**`chat_completion.ts` (DeepSeek dialect)** — ✅ Safe. Only `reasoning_content` is processed, which is always human-readable for DeepSeek.

**`response.ts`** — ⚠️ **ISSUE**: This is the gap. The code does NOT check `encrypted_content` before converting `reasoning_text` deltas to `ThinkingPart`. When streaming (`response.reasoning_text.delta`), any text delta is emitted as `{ type: 'thinking', content: delta.text }` without checking if the parent item is encrypted. Similarly in `fromResponseReasoningItem`, `encrypted_content` is not checked.

However, in practice for the official OpenAI API, when `encrypted_content` is present, the stream does NOT emit `response.reasoning_text.delta` events — only `summary` events are emitted. So the streaming path is safe for OpenAI. But third-party Response-compatible APIs might behave differently.

**Bottom line**: The ONLY code path with risk is `response.ts`'s lack of `encrypted_content` check. All other transports/dialects are safe.

### 3.2 Provider-specific tool calls — opaque serialization

Both transports use `JSON.stringify(rest)` / `JSON.parse(part.data)` for provider-specific tool calls. This pattern:

1. **Works for same-provider round-trips** (the JSON round-trips correctly)
2. **Fails across providers** (the opaque JSON is meaningless to other APIs)
3. **Loses type safety** (no validation that the parsed JSON matches expected shapes)
4. **Is fragile to API changes** (new fields in the SDK type are included in `...rest` but may have different semantics when serialized)

### 3.3 `server_tool_use` round-trip asymmetry

`ServerToolUsePart` is used for both:
- Server-side tool calls (file_search, web_search, code_interpreter, mcp_call, etc.)
- Provider-specific tool calls without `call_id`

When converting back:
- For Response: provider-specific ones are reconstructed from `JSON.parse(part.data)`
- For Chat Completions: `server_tool_use` parts are not handled in `toResponseInputItems` — they become `input_text` fallback (losing structure)

This asymmetry means `server_tool_use` parts are effectively Response-only for round-tripping.

---

## 4. Summary Table

| # | File | Severity | Issue |
|---|------|----------|-------|
| 1.1 | response.ts | **Critical** | `fromResponseReasoningItem` drops `content` (only keeps `summary`) |
| 1.2 | response.ts | **Critical** | No check for `encrypted_content` on reasoning items |
| 2.1 | chat_completion.ts | ~~Critical~~ **Resolved ✅** | Custom tool calls now round-trip correctly via `providerSpecific` marker |
| 2.2 | chat_completion.ts | **Critical** | Reasoning content silently dropped without dialect resolver |
| 1.3 | response.ts | **High** | Unsafe `JSON.parse` in `toItemFromServerToolUse` |
| 1.4 | response.ts | **High** | Lossy JSON round-trip for structured function_call_output arrays |
| 1.5 | response.ts | **High** | `ServerToolUsePart` with undefined `id` when no `call_id` |
| 2.3 | chat_completion.ts | **High** | Legacy `function_call` creates ToolCallPart with empty `id` |
| 1.6 | response.ts | **Medium** | Silent try-catch in `toToolResult` swallows errors |
| 1.7 | response.ts | **Medium** | Loose `EasyInputMessage` type guard (necessary for simplified format) |
| 1.8 | response.ts | **Medium** | Type mismatch in `reasoning_summary_text.delta` merge |
| 2.4 | chat_completion.ts | **Medium** | Tool message split may reorder content semantically |
| 2.5 | chat_completion.ts | **Medium** | No encrypted reasoning guard in core transport |
| 2.6 | chat_completion.ts | **Low** | `getFromToolMessageParam` now stringifies array content — loses per-part structure |
| 3.1 | Both | — | No explicit encrypted/pseudonymized thinking marker |
| 3.2 | Both | — | Provider-specific tool calls use opaque JSON |
| 3.3 | Both | — | `server_tool_use` round-trip is Response-only |
