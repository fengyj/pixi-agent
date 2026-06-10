# SDK ↔ ContentPart Conversion Mapping Reference

This document maps the data types between each LLM provider SDK's raw message format and the
framework's internal `ContentPart` / `SessionMessage` types. For each SDK there are two tables:
- **SDK type → ContentPart**: decoding a raw API message into `SessionMessage` parts.
- **ContentPart → SDK type**: encoding a `SessionMessage` back to a raw API message.

Where a cell says **DROPPED**, the converter silently discards the data. For cross-SDK
round-trips through `SessionMessage` this means information loss is possible. The document
also notes cases where the mapping is lossy (e.g. citations simplified, complex structures
serialised as JSON text blocks).

---

## 1. OpenAI Chat Completions

### 1.1 SDK type → ContentPart

| SDK type (source) | ContentPart | Notes |
|---|---|---|
| `{ type:'text', text }` – user content array | `TextPart` | One-to-one |
| `{ type:'text', text }` – assistant content (array or string) | `TextPart` | When content is a plain string no citations are created |
| `{ type:'refusal', refusal }` – assistant content array | `RefusalPart` | |
| `{ type:'image_url', image_url:{url} }` – user | `ImagePart` | `data:image/…;base64,…` ⇒ `sourceType:'base64'`, otherwise `'url'` |
| `{ type:'input_audio', input_audio:{data,format} }` – user | `AudioPart` | `mimeType` = `audio/{format}` (`wav` / `mp3`) |
| `{ type:'file', file:{file_data?,file_id?,filename?} }` – user | `DocumentPart` | `file_data` ⇒ `sourceType:'base64'`; `file_id` ⇒ `sourceType:'file_id'` |
| `tool_calls[]` entry `{type:'function',id,function:{name,arguments}}` – assistant | `ToolCallPart` | `arguments` is already a JSON string |
| `tool_calls[]` entry with non-`function` type – assistant | `ToolCallPart` | `providerSpecific: COMPLETIONS`; remaining fields serialised to `arguments` |
| `function_call` (legacy) – assistant | `ToolCallPart` | `id:''`, name & arguments preserved |
| `annotations[]` (`url_citation`) – assistant | `TextPart.citations` | Mapped to `CitationWebLocation`; `extra.rawCitationType:'url_citation'` |
| `refusal` (top-level) – assistant *string* content | `RefusalPart` | Only processed when `content` is a string; **lost when `content` is an array** |
| Tool message `{tool_call_id, content}` | `ToolResultPart` | `id` = `tool_call_id`; array content is `JSON.stringify`-ed |
| Function message (legacy) `{name, content}` | `ToolCallPart` (role: `assistant`) | Legacy fallback |
| `developer` / `system` role | `throw` | Unsupported – throws `invalidMessage` |

### 1.2 ContentPart → SDK type

| ContentPart | SDK type (target) | Notes |
|---|---|---|
| `TextPart` – user | `{type:'text', text}` | |
| `TextPart` – assistant | `{type:'text', text}` or plain string | **Citations lost** when `toAssistantContent` simplifies to a single string (array content cannot carry annotations) |
| `RefusalPart` – assistant | `{type:'refusal', refusal}` | |
| `ImagePart` – user | `{type:'image_url', image_url:{url}}` | base64 ⇒ data URL constructed; `file_id` sourceType ⇒ **DROPPED** |
| `AudioPart` – user | `{type:'input_audio', input_audio:{data,format}}` | `format` parsed from mimeType; `url` sourceType ⇒ **DROPPED** |
| `DocumentPart` – user | `{type:'file', file:{…}}` | sourceType drives which fields are set |
| `ToolCallPart` | `{type:'function', id, function:{name,arguments}}` (in `tool_calls[]`) | Arguments must be valid JSON |
| `ToolCallPart` with `providerSpecific:COMPLETIONS` | `tool_calls[]` entry with original non-function type | Preserved as-is |
| `ToolResultPart` | `{role:'tool', tool_call_id, content}` | Content may be string or array of `{type:'text',text}` |
| `ThinkingPart` | **DROPPED** | Not representable in Chat Completions |
| `ServerToolUsePart` | 📝 text block: `Tool use: {name} with data {data}` | Demoted via shared `ContentParts.serverToolUseFallbackText` |
| `AudioPart` / `VideoPart` / `DocumentPart` / `ImagePart` – assistant | 📝 text block: `JSON.stringify(rest)` | Demoted via shared `ContentParts.mediaFallbackText` |
| `VideoPart` – user | 📝 `{type:'text', text}` | Demoted to text via `ContentParts.mediaFallbackText` |

---

## 2. Anthropic Messages

### 2.1 SDK type → ContentPart

| SDK type (source) | ContentPart | Notes |
|---|---|---|
| `{type:'text', text, citations?}` | `TextPart` | Citations mapped: `char_location`/`page_location`/`content_block_location` ⇒ `others_location`; `search_result_location` ⇒ `others_location`; `web_search_result_location` ⇒ `web_location` |
| `{type:'thinking', thinking, signature}` | `ThinkingPart` | |
| `{type:'redacted_thinking'}` | (empty array) | Content unreadable & provider-specific – **dropped on purpose** |
| `{type:'image', source:{type:'base64'\|'url',…}}` | `ImagePart` | `sourceType:'base64'` or `'url'` |
| `{type:'document', source:{type:'base64',…}}` | `DocumentPart` | |
| `{type:'document', source:{type:'text',…}}` | `TextPart[]` | Document wrapper lost; content flattened to text parts |
| `{type:'document', source:{type:'url',…}}` | `DocumentPart` | |
| `{type:'document', source:{type:'content',…}}` | `TextPart[]` or `ImagePart[]` | Document wrapper lost; children flattened |
| `{type:'tool_use'}` with `caller.type==='direct'` (or no caller) | `ToolCallPart` | `input` is `JSON.stringify`-ed |
| `{type:'tool_use'}` with non-direct caller | `ServerToolUsePart` | `providerSpecific:ANTHROPIC`; `data` = JSON of block − `name` |
| `{type:'server_tool_use'}` with `caller.type==='direct'` | `ToolCallPart` | `providerSpecific:ANTHROPIC` |
| `{type:'server_tool_use'}` with non-direct caller | `ServerToolUsePart` | `providerSpecific:ANTHROPIC` |
| `{type:'tool_result', tool_use_id, content}` | `ToolResultPart` | Content may be string or JSON-stringified parts |
| `{type:'container_upload', file_id}` | `DocumentPart` | `sourceType:'file_id'` |
| `{type:'tool_reference', tool_name}` | `TextPart` | Content = `JSON.stringify({tool_name, type})` |
| `{type:'search_result', source, title, content}` | `TextPart` | Content = `JSON.stringify({source,title,content,type})` |
| `bash_code_execution_tool_result` | `ToolResultPart` | `providerSpecific:ANTHROPIC`; result = JSON of block − `tool_use_id` & `type` |
| `code_execution_tool_result` | `ToolResultPart` | Same pattern |
| `text_editor_code_execution_tool_result` | `ToolResultPart` | Same pattern |
| `tool_search_tool_result` | `ToolResultPart` | Same pattern |
| `web_fetch_tool_result` | `ToolResultPart` | Same pattern |
| `web_search_tool_result` | `ToolResultPart` | Same pattern |

### 2.2 ContentPart → SDK type

| ContentPart | SDK type (target) | Notes |
|---|---|---|
| `TextPart` | `{type:'text', text, citations?}` | Citations: `web_location` ⇒ `web_search_result_location`; `file_location` / `others_location` ⇒ **DROPPED** |
| `RefusalPart` | `{type:'text', text}` | Mapped to a text block; `refusal` type lost |
| `ThinkingPart` | `{type:'thinking', thinking, signature}` | |
| `ImagePart` (base64, jpeg/png/gif/webp) | `{type:'image', source:{type:'base64',…}}` | |
| `ImagePart` (url) | `{type:'image', source:{type:'url',…}}` | |
| `ImagePart` (file_id) | 📝 text block: `JSON.stringify(rest)` | Fallback via shared `ContentParts.mediaFallbackText` |
| `ImagePart` (base64, unknown mimeType) | 📝 text block | Same fallback |
| `DocumentPart` (base64, text/plain) | `{type:'document', source:{type:'text',…}}` | |
| `DocumentPart` (base64, application/pdf) | `{type:'document', source:{type:'base64',…}}` | |
| `DocumentPart` (url) | `{type:'document', source:{type:'url',…}}` | |
| `DocumentPart` (file_id) | 📝 text block: `JSON.stringify(rest)` | Fallback via shared `ContentParts.mediaFallbackText` |
| `DocumentPart` (base64, unknown mimeType) | 📝 text block | Same fallback |
| `ToolCallPart` | `{type:'tool_use', id, name, input}` | `arguments` parsed as JSON |
| `ToolCallPart` with `providerSpecific:ANTHROPIC` | Reconstructed `ServerToolUseBlockParam` | `data` parsed & spread, `name` overlaid |
| `ToolResultPart` | `{type:'tool_result', tool_use_id, content}` | Result may be parsed to array of content blocks |
| `ToolResultPart` with `providerSpecific:ANTHROPIC` | Reconstructed specific result (e.g. `web_search_tool_result`) | `name` pattern `xxx_tool_result` reconstructs type |
| `ServerToolUsePart` with `providerSpecific:ANTHROPIC` | Reconstructed block | `data` parsed & spread |
| `ServerToolUsePart` without ANTHROPIC marker | `{type:'text', text}` | Fallback: `"Tool use: {name} with data {data}"` |
| `AudioPart` | `{type:'text', text}` | Fallback: `JSON.stringify({audio, type})` |
| `VideoPart` | `{type:'text', text}` | Fallback: `JSON.stringify({video, type})` |

---

## 3. OpenAI Responses

### 3.1 SDK type → ContentPart

| SDK type (source) | ContentPart | Notes |
|---|---|---|
| `input_text` | `TextPart` | |
| `output_text` | `TextPart` | `annotations` ⇒ `citations` (url_citation, file_citation, container_file_citation, file_path) |
| `refusal` | `RefusalPart` | |
| `input_image` (image_url / file_id) | `ImagePart` | sourceType derived from fields present |
| `input_file` (file_id / file_url / file_data) | `DocumentPart` | sourceType derived from fields present |
| `function_call` | `ToolCallPart` | `id` = `call_id` |
| `function_call_output` | `ToolResultPart` | Structured output (array) ⇒ JSON stringified |
| `reasoning` | `ThinkingPart` | Only `summary` text extracted; structured summary entries lost |
| `image_generation_call` | `ImagePart` | `sourceType:'base64'`, `mimeType:'image/png'` |
| `computer_call` (has `call_id`) | `ToolCallPart` | `providerSpecific:RESPONSE` |
| `computer_call` (no `call_id`) | `ServerToolUsePart` | `providerSpecific:RESPONSE` |
| `custom_tool_call` (has `call_id`) | `ToolCallPart` | `providerSpecific:RESPONSE` |
| `tool_search_call` (has `call_id`) | `ToolCallPart` | `providerSpecific:RESPONSE` |
| `local_shell_call` / `shell_call` / `apply_patch_call` (has `call_id`) | `ToolCallPart` | `providerSpecific:RESPONSE` |
| `local_shell_call` / `shell_call` / `apply_patch_call` (no `call_id`) | `ServerToolUsePart` | `providerSpecific:RESPONSE` |
| `file_search_call` / `web_search_call` / `code_interpreter_call` | `ServerToolUsePart` | `providerSpecific:RESPONSE` |
| `mcp_call` / `mcp_list_tools` | `ServerToolUsePart` | `providerSpecific:RESPONSE` |
| Specific tool outputs (has `call_id`) | `ToolResultPart` | `providerSpecific:RESPONSE`; result = JSON of block − `call_id` & `type` |
| Specific tool outputs (no `call_id`) | `ServerToolUsePart` | `providerSpecific:RESPONSE` |
| `message` (string content) | `TextPart[]` | Single text part |
| `message` (mixed content) | `TextPart[]` / `ImagePart[]` / `DocumentPart[]` | Flattened |
| `mcp_approval_request` / `mcp_approval_response` | **DROPPED** | |
| `compaction` / `compaction_trigger` / `item_reference` | **DROPPED** | |

### 3.2 ContentPart → SDK type

| ContentPart | SDK type (target) | Notes |
|---|---|---|
| `TextPart` – user | `input_text` (inside a message envelope) | |
| `TextPart` – assistant | `output_text` (inside a message envelope) | Citations ⇒ annotations (url/file/container_file/file_path) |
| `RefusalPart` – assistant | `refusal` (inside a message envelope) | |
| `ImagePart` – user | `input_image` | sourceType ⇒ `image_url` or `file_id` |
| `ImagePart` – assistant (base64 only) | `image_generation_call` | Other sourceTypes → 📝 `output_text` via `ContentParts.mediaFallbackText` |
| `DocumentPart` – user | `input_file` | sourceType ⇒ `file_url` / `file_data` / `file_id` |
| `ToolCallPart` | `function_call` | `call_id` = `id` |
| `ToolCallPart` with `providerSpecific:RESPONSE` | Reconstructed specific call (e.g. `computer_call`) | Spread from stored JSON |
| `ToolResultPart` | `function_call_output` | Structured output if result parses as array of content parts |
| `ToolResultPart` with `providerSpecific:RESPONSE` | Reconstructed specific output (e.g. `computer_call_output`) | Spread from stored JSON |
| `ThinkingPart` | `reasoning` | Content ⇒ `summary_text` |
| `ServerToolUsePart` with `providerSpecific:RESPONSE` | Reconstructed block (e.g. `web_search_call`) | Spread from stored JSON |
| `ServerToolUsePart` without RESPONSE – user/tool | `input_text` | Fallback: `ContentParts.serverToolUseFallbackText` |
| `ServerToolUsePart` without RESPONSE – assistant | `output_text` | Fallback: `ContentParts.serverToolUseFallbackText` |
| `AudioPart` – assistant | 📝 `output_text`: `JSON.stringify(rest)` | Fallback via `ContentParts.mediaFallbackText` |
| `VideoPart` – assistant | 📝 `output_text` | Same fallback |
| `DocumentPart` – assistant | 📝 `output_text` | Same fallback |

---

## Cross-SDK data flow summary

All cross-SDK conversions go through `SessionMessage` as an intermediate representation:

```
SDK-A raw → SessionMessage → SDK-B raw
```

| Scenario | Key behaviour |
|---|---|
| Anthropic `tool_use` → ChatCompletion | `ToolCallPart` → `tool_calls[]` entry. Works correctly. |
| Anthropic `thinking` → ChatCompletion | Thinking is **dropped** (ChatCompletion has no thinking concept). |
| Anthropic specific tool results → ChatCompletion | `ToolResultPart` → tool message. Provider-specific fields (e.g. `stdout`) dropped; only `result` string kept. |
| ChatCompletion `function_call` → Anthropic | Works. Empty `id` carried through (Anthropic may auto-generate). |
| ChatCompletion `audio`/`image` → Anthropic | Audio ⇒ text block (JSON). Image ⇒ image block. |
| Response `reasoning` → Anthropic `thinking` | Works. Structured summary flattened to single text. |
| Response `function_call` → ChatCompletion | Works. |
| Response `image_generation_call` → ChatCompletion | Image carried as base64 `ImagePart`. |
| Anthropic `thinking` → Response `reasoning` | Works. |
| Any SDK → Session → Another SDK with `server_tool_use` | ChatCompletion demotes them to text. Anthropic & Response reconstruct them with `providerSpecific` markers. If the marker is missing, the converter falls back to a text block. |
