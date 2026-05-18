# PixiAgent 可观测性方案

本文档描述如何在 `PixiAgent` 中使用 `Observation` 模块添加结构化日志和分布式追踪。

---

## 目录

1. [总体原则](#1-总体原则)
2. [Logger：日志方案](#2-logger日志方案)
3. [Tracer：追踪方案](#3-tracer追踪方案)
4. [各方法的插桩点](#4-各方法的插桩点)
5. [属性命名规范](#5-属性命名规范)
6. [完整代码示例](#6-完整代码示例)

---

## 1. 总体原则

- **Logger 和 Tracer 实例在模块级别创建**，不在构造函数里创建。pino child logger 和 OTel tracer 都是轻量对象，可以安全地放在模块顶部。
- **每个 `PixiAgent` 实例持有一个绑定了 session/thread 上下文的 child logger**，这样该实例输出的所有日志都自动携带 `sessionId` 和 `threadId`，不需要每次手动传。
- **Trace span 与方法调用一一对应**，形成层级结构，便于在 Jaeger/Tempo 等工具里观察完整的 agent 执行链路。
- **日志级别选择**：
  - `error`：不可恢复的错误（未知 role、达到 maxIterations 异常）
  - `warn`：可恢复但值得关注的情况（无效 pending message、已在运行时收到新 execute 调用）
  - `info`：每次 LLM 调用完成（含 token 用量）、interrupt 发起
  - `debug`：循环开始/结束、每次迭代、工具调用细节

---

## 2. Logger：日志方案

### 2.1 模块级 logger

在 `agent.ts` 文件顶部创建一个模块级 logger，scope 名称为 `'agent'`：

```typescript
import { Observation } from './observation';

const log = Observation.getLogger('agent');
```

### 2.2 实例级 child logger

在 `PixiAgent` 构造函数中，基于 session/thread 上下文创建 child logger，绑定为实例属性：

```typescript
export class PixiAgent {
  private readonly log: ReturnType<typeof log.child>;

  constructor(
    public sessionThread: SessionThread,
    public toolRegistry: ToolRegistry,
    public options?: PixiAgentOptions,
  ) {
    this.log = log.child({
      sessionId: sessionThread.session.sessionId,
      threadId: sessionThread.threadInfo.threadId,
    });
    // ...
  }
}
```

这样 `this.log.info(...)` 输出的每条日志都会自动带上 `sessionId` 和 `threadId`。

### 2.3 关键日志点一览

| 方法 | 级别 | 时机 | 关键字段 |
|------|------|------|----------|
| `execute()` | `debug` | 进入时 | `model`, `pendingCount` |
| `execute()` | `warn` | 已在运行，跳过 | `model` |
| `execute()` | `warn` | maxIterations 达到上限 | `iterations`, `maxIterations` |
| `execute()` | `error` | 循环抛出未知错误 | `err` |
| `execute()` | `debug` | 正常结束（队列清空/被 abort） | `iterations`, `reason` |
| `interrupt()` | `info` | 发出中断 | `reason` |
| `interrupt()` | `debug` | 重复调用，已中断 | — |
| `consumePendingMessages()` | `warn` | 发现空/无效 pending message | `role`, `pendingMessageIds` |
| `executeLLMRequest()` | `debug` | 发起 LLM 请求前 | `model`, `role`, `historyCount` |
| `executeLLMRequest()` | `info` | LLM 响应收到后 | `model`, `inputTokens`, `outputTokens`, `hasToolCalls` |
| `executeToolCallRequest()` | `debug` | 工具调用开始 | `toolNames`, `parallel` |
| `executeToolCallRequest()` | `debug` | 工具调用结束 | `toolNames`, `errorCount` |

---

## 3. Tracer：追踪方案

### 3.1 模块级 tracer

```typescript
const tracer = Observation.getTracer('pixiagent.agent');
```

### 3.2 Span 层级结构

```
execute (root span)
│  attributes: gen_ai.model, thread.id, session.id
│
└── agent.iteration (每次循环一个 span)
│     attributes: iteration
│
├── agent.llm_request (executeLLMRequest)
│     attributes: gen_ai.model, gen_ai.operation.name="chat",
│                 gen_ai.usage.input_tokens, gen_ai.usage.output_tokens,
│                 agent.has_tool_calls
│
└── agent.tool_calls (executeToolCallRequest)
      attributes: agent.tool_names, agent.parallel_tool_calls
      │
      ├── agent.tool_execute (每个工具一个 span)
      │     attributes: gen_ai.tool.name
      │
      └── agent.tool_execute
            attributes: gen_ai.tool.name
```

`execute` 是顶层 span，覆盖整个执行周期（可能跨多轮 LLM 调用）；每一次迭代再套一个 `agent.iteration` span，使得 Trace 视图中可以清楚看到 agent 执行了几轮。

### 3.3 错误处理

发生异常时需要在 span 上记录异常并设置错误状态，然后再 rethrow：

```typescript
import { SpanStatusCode } from '@opentelemetry/api';

span.recordException(err as Error);
span.setStatus({ code: SpanStatusCode.ERROR, message: (err as Error).message });
throw err;
```

---

## 4. 各方法的插桩点

### 4.1 `execute()`

```typescript
public async execute(modelOptions: ModelOptions, input: PendingMessage): Promise<void> {
  this.sessionThread.addPendingMessage(input);

  if (this.isRunning) {
    this.log.warn({ model: modelOptions.model }, 'execute() called while already running; input queued');
    return;
  }

  this.log.debug({ model: modelOptions.model, pendingCount: this.sessionThread.getPendingMessages().length }, 'agent execute start');

  return tracer.startActiveSpan('agent.execute', async (span) => {
    span.setAttribute('gen_ai.model', modelOptions.model);
    span.setAttribute('thread.id', this.sessionThread.threadInfo.threadId);
    span.setAttribute('session.id', this.sessionThread.session.sessionId);

    this.isRunning = true;
    modelOptions = PixiAgent.resolveApiModeAndBaseUrl(modelOptions);
    // ...

    try {
      let iterations = 0;
      while (!this.abortController.signal.aborted) {
        // ...
        await tracer.startActiveSpan('agent.iteration', async (iterSpan) => {
          iterSpan.setAttribute('agent.iteration', iterations);
          try {
            await this.consumePendingMessages(modelOptions);
          } finally {
            iterSpan.end();
          }
        });
        iterations++;
      }

      const reason = this.abortController.signal.aborted ? 'aborted' : 'queue_empty';
      this.log.debug({ iterations, reason }, 'agent execute finished');
    } catch (err) {
      span.recordException(err as Error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: (err as Error).message });
      this.log.error({ err }, 'agent execute error');
      throw err;
    } finally {
      this.isRunning = false;
      span.end();
    }
  });
}
```

### 4.2 `interrupt()`

```typescript
public interrupt(reason?: string): void {
  if (this.abortController.signal.aborted) {
    this.log.debug('interrupt() called but already aborted, ignoring');
    return;
  }
  this.log.info({ reason }, 'agent interrupted');
  this.abortController.abort(new Error(reason ?? 'User interrupted'));
}
```

### 4.3 `consumePendingMessages()` - 无效消息告警

```typescript
if (/* 空内容检测 */) {
  this.log.warn(
    { role: sessionMessage.role, pendingMessageIds: pendingMessages.map(m => m.pendingMessageId) },
    'dropping empty pending message'
  );
  this.sessionThread.removePendingMessage(...);
  return;
}
```

### 4.4 `executeLLMRequest()`

```typescript
private async executeLLMRequest(modelOptions: ModelOptions, sessionMessage: SessionMessage): Promise<void> {
  return tracer.startActiveSpan('agent.llm_request', async (span) => {
    span.setAttribute('gen_ai.model', modelOptions.model);
    span.setAttribute('gen_ai.operation.name', 'chat');

    this.log.debug(
      { model: modelOptions.model, role: sessionMessage.role, historyCount: this.sessionThread.threadMessages.length },
      'LLM request start'
    );

    try {
      // ... transport.generate() ...

      const hasToolCalls = /* ... */ false;
      span.setAttribute('gen_ai.usage.input_tokens', response.usage?.inputTokens ?? 0);
      span.setAttribute('gen_ai.usage.output_tokens', response.usage?.outputTokens ?? 0);
      span.setAttribute('agent.has_tool_calls', hasToolCalls);

      this.log.info(
        {
          model: modelOptions.model,
          inputTokens: response.usage?.inputTokens,
          outputTokens: response.usage?.outputTokens,
          hasToolCalls,
        },
        'LLM call completed'
      );
    } catch (err) {
      span.recordException(err as Error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: (err as Error).message });
      throw err;
    } finally {
      span.end();
    }
  });
}
```

### 4.5 `executeToolCallRequest()`

```typescript
private async executeToolCallRequest(modelOptions: ModelOptions, sessionMessage: SessionMessage): Promise<void> {
  // ...
  const toolNames = toolCalls.map(tc => tc.name);
  const parallel = modelOptions.parallelToolCalls ?? true;

  this.log.debug({ toolNames, parallel }, 'tool calls start');

  return tracer.startActiveSpan('agent.tool_calls', async (span) => {
    span.setAttribute('agent.tool_names', toolNames.join(','));
    span.setAttribute('agent.parallel_tool_calls', parallel);

    try {
      if (parallel) {
        const results = await Promise.all(
          toolCalls.map(tc =>
            tracer.startActiveSpan('agent.tool_execute', async (toolSpan) => {
              toolSpan.setAttribute('gen_ai.tool.name', tc.name);
              try {
                return await this.toolRegistry.execute(tc, { signal: this.abortController.signal });
              } catch (err) {
                toolSpan.recordException(err as Error);
                toolSpan.setStatus({ code: SpanStatusCode.ERROR });
                throw err;
              } finally {
                toolSpan.end();
              }
            })
          )
        );
        // ...
      }

      const errorCount = results.filter(r => r.isError).length;
      this.log.debug({ toolNames, errorCount }, 'tool calls completed');
    } catch (err) {
      span.recordException(err as Error);
      span.setStatus({ code: SpanStatusCode.ERROR });
      throw err;
    } finally {
      span.end();
    }
  });
}
```

---

## 5. 属性命名规范

遵循 [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `gen_ai.model` | string | 模型名称，如 `"gpt-4o"` |
| `gen_ai.operation.name` | string | 操作类型，LLM 调用用 `"chat"` |
| `gen_ai.usage.input_tokens` | int | 输入 token 数 |
| `gen_ai.usage.output_tokens` | int | 输出 token 数 |
| `gen_ai.tool.name` | string | 工具名称 |
| `thread.id` | string | SessionThread 的 threadId |
| `session.id` | string | Session 的 sessionId |
| `agent.iteration` | int | 当前迭代次数（0-based） |
| `agent.has_tool_calls` | bool | LLM 响应是否包含工具调用 |
| `agent.tool_names` | string | 逗号分隔的工具名列表 |
| `agent.parallel_tool_calls` | bool | 工具调用是否并行执行 |

---

## 6. 完整代码示例

将以上所有内容拼在一起，`agent.ts` 的顶部引入部分如下：

```typescript
import { Observation } from './observation';
import { SpanStatusCode } from '@opentelemetry/api';

// 模块级，只创建一次
const log = Observation.getLogger('agent');
const tracer = Observation.getTracer('pixiagent.agent');

export class PixiAgent {
  private readonly log: ReturnType<typeof log.child>;

  constructor(/* ... */) {
    this.log = log.child({
      sessionId: sessionThread.session.sessionId,
      threadId: sessionThread.threadInfo.threadId,
    });
    // ...
  }

  // 各方法见上方第 4 节
}
```

> **注意**：`SpanStatusCode` 来自 `@opentelemetry/api`，该包已作为 `Observation` 模块的依赖存在，不需要额外安装。
