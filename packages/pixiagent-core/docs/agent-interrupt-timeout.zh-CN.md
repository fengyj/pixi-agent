# PixiAgent 中断与超时实现指南

这份文档面向 TypeScript/JavaScript 初学者，目标是回答 4 个问题：

1. 如何为 `PixiAgent.interrupt()` 实现中断 API。
2. 如何让 `PixiAgent.execute()` 支持超时。
3. 在定义 tool 的 `handler` 和 `funcChecker` 时，如何处理中断和超时。
4. `ToolRegistry.execute()` 里是否应该加中断和超时逻辑。

---

## 1. 先讲清楚：AbortController / AbortSignal 是什么

可以把它理解成一套“停止开关”机制。

- `AbortController`：开关控制器，谁持有它，谁可以按下“停止”。
- `AbortSignal`：只读信号，传给正在执行的任务，任务可以监听这个信号决定何时停止。

最小示例：

```ts
const controller = new AbortController();
const signal = controller.signal;

signal.addEventListener('abort', () => {
  console.log('任务被取消了，原因:', signal.reason);
});

controller.abort(new Error('User interrupted'));
```

关键点：

- `abort()` 只会触发一次，后续再触发没有意义。
- `signal.aborted` 可以随时检查。
- `signal.reason` 记录取消原因（Node.js 新版本可用）。

### 1.1 超时和中断本质上是一回事

“超时”可以看成“系统触发的中断”。

通常做法：

- 创建一个“超时控制器”，到时 `abort(new Error('Timeout ...'))`。
- 把它和“用户中断控制器”合并成一个 signal，传给 LLM 请求和 tool 执行。

---

## 2. 当前代码现状（基于现有实现）

你的代码已经有这些基础：

- `PixiAgent.interrupt()` 目前是空实现。
- transport 层已经支持 `ModelRequestOptions`：包含 `signal` 和 `timeout`。
- `ToolCallOptions` 也已有 `signal` 和 `timeout` 字段。
- `ToolRegistry.execute()` 当前没有真正实现“超时包装”和“中断包装”。

所以方向是正确的，只是还没把“控制流”真正串起来。

---

## 3. 设计建议：把中断做成“单次运行上下文”

建议把一次 `execute()` 看作一次 run，并创建 run 级别的上下文：

```ts
type AgentRunOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;     // 整个 execute 的总超时
  llmTimeoutMs?: number;  // 单次 LLM 请求超时
  toolTimeoutMs?: number; // 单个 tool call 超时
};
```

在 `PixiAgent` 里新增状态：

```ts
private activeRunController?: AbortController;
private activeRunSignal?: AbortSignal;
```

含义：

- `interrupt()` 只需要 `this.activeRunController?.abort(...)`。
- `execute()` 内部所有异步任务都使用 `activeRunSignal`（或它的子 signal）。

---

## 4. 如何实现 PixiAgent.interrupt()

目标行为：

- 如果当前有 run 在执行，立即触发 abort。
- 没有 run 时，安全 no-op。

建议语义：

```ts
public async interrupt(reason?: unknown): Promise<void> {
  this.activeRunController?.abort(reason ?? new Error('Agent interrupted'));
}
```

你还可以定义一个统一错误类型（可选）：

```ts
class AgentInterruptedError extends Error {
  constructor(message = 'Agent interrupted') {
    super(message);
    this.name = 'AgentInterruptedError';
  }
}
```

---

## 5. 如何让 PixiAgent.execute() 支持超时

推荐分 3 层：

1. run 总超时（保护整个 while 循环）。
2. LLM 请求超时（保护 `transport.generate`）。
3. tool call 超时（保护 `toolRegistry.execute` / `tool.handler`）。

### 5.1 先准备两个通用工具函数

```ts
function createTimeoutSignal(timeoutMs: number, label: string): {
  signal: AbortSignal;
  cleanup: () => void;
} {
  const c = new AbortController();
  const timer = setTimeout(() => {
    c.abort(new Error(`${label} timeout after ${timeoutMs}ms`));
  }, timeoutMs);
  return {
    signal: c.signal,
    cleanup: () => clearTimeout(timer),
  };
}

function combineSignals(signals: Array<AbortSignal | undefined | null>): AbortSignal {
  const valid = signals.filter(Boolean) as AbortSignal[];
  if (valid.length === 0) {
    return new AbortController().signal;
  }
  if (valid.length === 1) {
    return valid[0];
  }
  const c = new AbortController();
  const onAbort = (s: AbortSignal) => {
    c.abort(s.reason ?? new Error('Aborted'));
  };
  for (const s of valid) {
    if (s.aborted) {
      onAbort(s);
      break;
    }
    s.addEventListener('abort', () => onAbort(s), { once: true });
  }
  return c.signal;
}
```

说明：

- `createTimeoutSignal` 负责超时触发。
- `combineSignals` 负责“任意一个 signal 取消，就整体取消”。

### 5.2 execute 的控制流建议

伪代码：

```ts
public async execute(modelOptions: ModelOptions, input: PendingMessage, runOptions?: AgentRunOptions): Promise<void> {
  const runController = new AbortController();
  this.activeRunController = runController;

  const runTimeout = runOptions?.timeoutMs
    ? createTimeoutSignal(runOptions.timeoutMs, 'execute')
    : undefined;

  const runSignal = combineSignals([
    runController.signal,
    runOptions?.signal,
    runTimeout?.signal,
  ]);
  this.activeRunSignal = runSignal;

  try {
    this.sessionThread.addPendingMessage(input);

    while (this.sessionThread.getPendingMessages().length > 0) {
      if (runSignal.aborted) throw runSignal.reason ?? new Error('Run aborted');

      const pendingMessages = PixiAgent.peekPendingMessagesToExecute(this.sessionThread);
      const sessionMessage = PixiAgent.convertPendingMessages(this.sessionThread, pendingMessages);

      if (sessionMessage.role === 'user' || sessionMessage.role === 'tool') {
        await this.executeLLMRequest(modelOptions, sessionMessage, runOptions, runSignal);
      } else {
        await this.executeToolCallRequest(modelOptions, sessionMessage, runOptions, runSignal);
      }

      this.sessionThread.removePendingMessage(pendingMessages.map((m) => m.pendingMessageId));
    }
  } finally {
    runTimeout?.cleanup();
    this.activeRunController = undefined;
    this.activeRunSignal = undefined;
  }
}
```

### 5.3 LLM 请求超时

在 `executeLLMRequest` 中：

- 创建 llm 局部 timeout signal。
- 和 runSignal 合并。
- 传给 `transport.generate(..., requestOptions)`。

```ts
const llmTimeout = runOptions?.llmTimeoutMs
  ? createTimeoutSignal(runOptions.llmTimeoutMs, 'llm request')
  : undefined;

const llmSignal = combineSignals([runSignal, llmTimeout?.signal]);

const response = await transport.generate(
  modelOptions,
  [...historyMessages, rawMessage],
  {},
  {
    signal: llmSignal,
    timeout: runOptions?.llmTimeoutMs,
  },
);
```

---

## 6. tool.handler / funcChecker 如何实现中断与超时

### 6.1 对 handler 的建议

`handler` 可能是慢操作（网络、IO、外部进程），必须支持中断和超时。

最常见写法：

```ts
const searchTool = defineTool({
  name: 'search',
  description: 'Search data',
  schema: z.object({ query: z.string() }),
  funcChecker: () => true,
  handler: async (input, options) => {
    if (options?.signal?.aborted) {
      throw options.signal.reason ?? new Error('Tool aborted before start');
    }

    const res = await fetch(`https://example.com?q=${encodeURIComponent(input.query)}`, {
      signal: options?.signal ?? undefined,
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  },
});
```

如果底层 API 不支持 signal：

- 在 handler 内手动监听 `abort` 并提前 reject。
- 或者包一层 `Promise.race([realTask, abortPromise])`。

### 6.2 对 funcChecker 的建议

`funcChecker` 应该是：

- 快速、同步、纯检查。
- 不做长耗时操作。

典型检查：

- 必要环境变量是否存在。
- 本地依赖是否可用。

示例：

```ts
funcChecker: () => {
  return Boolean(process.env.MY_API_KEY);
}
```

不建议在 `funcChecker` 里处理中断/超时，因为它本来就应该是瞬时检查。

如果你确实需要远程检查可用性，建议把“检查动作”放进 handler 主流程里，并在 handler 里用 signal/timeout。

---

## 7. ToolRegistry.execute 里要不要加中断和超时逻辑？

结论：要加，而且应该加。

原因：

- `ToolRegistry.execute` 是所有 tool 调用的统一入口。
- 中断/超时属于“横切关注点”，适合在统一入口收口，避免每个 tool 重复写。
- 可统一错误格式（`isError: true` + 标准错误文本），便于上层处理。

### 7.1 建议职责分层

- `ToolRegistry.execute`：
  - 合并 signal。
  - 创建 tool 级 timeout。
  - 执行前检查 `funcChecker`。
  - 执行后包装结果和错误。
- `tool.handler`：
  - 真实业务逻辑。
  - 在可中断点响应 `options.signal`。

### 7.2 execute 里的参考逻辑

```ts
async execute(toolCall: ToolCallPart, options?: ToolCallOptions): Promise<ToolResultPart> {
  const tool = this.tools.get(toolCall.name);
  if (!tool) {
    return {
      type: 'tool_result',
      id: toolCall.id,
      name: toolCall.name,
      result: JSON.stringify({ error: `Tool ${toolCall.name} not found` }),
      isError: true,
    };
  }

  const timeoutHandle = options?.timeout
    ? createTimeoutSignal(options.timeout, `tool ${toolCall.name}`)
    : undefined;

  const signal = combineSignals([options?.signal, timeoutHandle?.signal]);

  try {
    if (signal.aborted) throw signal.reason ?? new Error('Tool aborted');

    if (!tool.funcChecker({ ...options, signal })) {
      return {
        type: 'tool_result',
        id: toolCall.id,
        name: toolCall.name,
        result: JSON.stringify({ error: `Tool ${toolCall.name} is currently unavailable` }),
        isError: true,
      };
    }

    const parsedArgs = typeof toolCall.arguments === 'string'
      ? JSON.parse(toolCall.arguments)
      : toolCall.arguments;
    const parsedInput = tool.schema.parse(parsedArgs);

    const result = await tool.handler(parsedInput, { ...options, signal });
    return {
      type: 'tool_result',
      id: toolCall.id,
      name: toolCall.name,
      result: JSON.stringify(result),
    };
  } catch (error) {
    return {
      type: 'tool_result',
      id: toolCall.id,
      name: toolCall.name,
      result: JSON.stringify({ error: (error as Error).message }),
      isError: true,
    };
  } finally {
    timeoutHandle?.cleanup();
  }
}
```

---

## 8. 并行 tool 调用时的中断行为

你当前支持 `parallelToolCalls`。建议语义如下：

- `parallelToolCalls = true`：
  - 同一个 runSignal 传给所有并行 tool。
  - 任意时刻发生中断，所有工具都应尽快停止。
- `parallelToolCalls = false`：
  - 串行执行，每步开始前检查 `signal.aborted`。

如果某个 tool 本身无法真正取消（例如第三方 SDK 不支持），至少要做到：

- 上层不再等待其结果（逻辑上视为已中断）。
- 记录日志，避免“静默卡住”。

---

## 9. 错误分类建议（便于上层处理）

建议至少区分：

- 用户中断：`AgentInterruptedError`
- 超时：`TimeoutError`
- 普通业务错误：`Error`

然后在 `ToolResultPart.result` 里保留结构化字段，例如：

```json
{
  "error": "tool search timeout after 5000ms",
  "errorType": "timeout"
}
```

这样 UI 层可以显示“已超时”而不是笼统“执行失败”。

---

## 10. 建议测试清单

最少补这些测试：

1. `interrupt()` 在 LLM 请求中触发，`execute()` 能快速结束。
2. `execute(timeoutMs)` 到时自动中断。
3. `llmTimeoutMs` 仅中断当前 LLM 请求。
4. `toolTimeoutMs` 能把超时映射到 `ToolResultPart.isError = true`。
5. 并行 tool 下，run 中断后所有 tool 都收到 `signal.aborted`。
6. 串行 tool 下，中断后不会继续执行后续 tool。

---

## 11. 一份最简落地顺序（建议按这个顺序改）

1. 先实现 `PixiAgent.interrupt()` + run 级 controller。
2. 在 `execute()` 注入 runSignal 和总超时。
3. 给 `executeLLMRequest()` 透传 signal/timeout。
4. 给 `executeToolCallRequest()` 透传 signal/timeout。
5. 在 `ToolRegistry.execute()` 增加统一超时/中断包装。
6. 最后补测试，先测串行，再测并行。

---

## 12. 结论（直接回答你的问题）

1. `PixiAgent.interrupt`：
   - 用 `AbortController` 最合适，内部保存当前 run 的 controller，`interrupt()` 触发 `abort()`。

2. `PixiAgent.execute` 支持超时：
   - 用 run 级 `timeoutMs` + 子阶段 `llmTimeoutMs/toolTimeoutMs`，都通过 signal 统一传递。

3. tool 的 `handler` 和 `funcChecker`：
   - `handler` 要响应 `options.signal`，并在慢操作中传递 signal。
   - `funcChecker` 应保持轻量同步检查，不应承担慢操作的超时/中断逻辑。

4. `ToolRegistry.execute` 是否加中断超时逻辑：
   - 应该加。这里是统一入口，最适合处理横切逻辑并统一错误返回格式。
