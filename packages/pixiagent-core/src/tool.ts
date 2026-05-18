import { z } from 'zod';
import { ToolCallPart, ToolResultPart } from './message';

export type ToolCallOptions = {
  signal?: AbortSignal | undefined | null;
  timeout?: number;
  maxRetries?: number;
  environment?: Record<string, unknown>; // optional environment variables to pass to the tool
};

export const ToolDefinitionSchema = z.object({
  name: z.string(),
  description: z.string(),
  parameters: z.record(z.string(), z.unknown()),
});

export type ToolDefinition = z.infer<typeof ToolDefinitionSchema>;

export type Tool<
  TInputSchema extends z.ZodObject<z.ZodRawShape> = z.ZodObject<z.ZodRawShape>,
  TOutput = unknown,
> = {
  definition: ToolDefinition;
  schema: TInputSchema;
  handler: (input: z.infer<TInputSchema>, options?: ToolCallOptions) => Promise<TOutput>;
  funcChecker: (options?: ToolCallOptions) => boolean; // A function to check if the tool can be called, e.g. check if the API key is set or if the user has permission to use the tool. Returns true if the tool can be called, false otherwise.
};

export function defineTool<
  TInputSchema extends z.ZodObject<z.ZodRawShape>,
  TOutput = unknown,
>(opts: {
  name: string;
  description: string;
  schema: TInputSchema;
  /**
   * The function of the tool
   * @param input the input of the tool
   * @param options the options for the tool call, including signal for abortion,
   *                timeout, maxRetries, and environment variables.
   *                The tool handler should respect the signal for abortion and timeout (
   *                unless the tool is designed to be non-interruptible and won't timeout),
   *                and can implement retry logic based on maxRetries.
   *                Environment variables can be used for the tool's execution environment.
   * @returns
   */
  handler: (input: z.infer<TInputSchema>, options?: ToolCallOptions) => Promise<TOutput>;
  /**
   * A function to check if the tool can be called, e.g. check if the API key is set
   * or if the user has permission to use the tool.
   * @param options the options for the tool call, including signal for abortion, 
   *                timeout, maxRetries, and environment variables.
   * @returns true if the tool can be called, false otherwise.
   */
  funcChecker: (options?: ToolCallOptions) => boolean;
}): Tool<TInputSchema, TOutput> {
  return {
    definition: {
      name: opts.name,
      description: opts.description,
      parameters: z.toJSONSchema(opts.schema),
    },
    schema: opts.schema,
    handler: opts.handler,
    funcChecker: opts.funcChecker, // todo: before returning the avaialbe tools from registry, need to check the funcChecker and filter out the tools that cannot be called.
  };
}

export type Toolset = {
  name: string;
  description: string;
  tools: Tool[];
  includes: string[]; // list of toolset names to include, supports nested inclusion
};

export class ToolRegistry {
  private toolsets = new Map<string, Toolset>();
  private tools = new Map<string, Tool>();

  registerToolset(toolset: Toolset): this {
    this.toolsets.set(toolset.name, toolset);
    for (const tool of toolset.tools) {
      this.registerTool(tool);
    }
    return this;
  }

  registerTool(tool: Tool): this {
    this.tools.set(tool.definition.name, tool);
    return this;
  }

  getDefinitions(activeToolsets?: string[]): ToolDefinition[] {
    if (!activeToolsets) {
      return Array.from(this.tools.values())
        .filter((t) => t.funcChecker())
        .map((t) => t.definition);
    }
    const sets = activeToolsets.map((n) => this.toolsets.get(n)!).filter(Boolean);

    return sets.flatMap((ts) =>
      ts.tools
        .filter((t) => t.funcChecker())
        .map((t) => t.definition),
    );
  }

  getTool(name: string): Tool | undefined {
    return this.tools.get(name);
  }

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

    try {
      if (!tool.funcChecker(options)) {
        return {
          type: 'tool_result',
          id: toolCall.id,
          name: toolCall.name,
          result: JSON.stringify({ error: `Tool ${toolCall.name} is currently unavailable` }),
          isError: true,
        };
      }
      const parsedInput = tool.schema.parse(toolCall.arguments);
      const result = await tool.handler(parsedInput, options);
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
    }
  }
}
