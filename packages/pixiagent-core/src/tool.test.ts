import { describe, expect, it } from 'vitest';
import { z } from 'zod';
import { defineTool } from './tool';

describe('defineTool', () => {
  it('creates a tool definition with JSON schema and executes handler', async () => {
    const echoTool = defineTool({
      name: 'echo',
      description: 'Echo input text',
      schema: z.object({ text: z.string() }),
      handler: async (input) => ({ echoed: input.text }),
      funcChecker: () => true,
    });

    expect(echoTool.definition.name).toBe('echo');
    expect(echoTool.definition.description).toBe('Echo input text');
    expect(echoTool.definition.parameters).toBeTypeOf('object');

    const result = await echoTool.handler({ text: 'hello' });
    expect(result).toEqual({ echoed: 'hello' });
    expect(echoTool.funcChecker()).toBe(true);
  });
});