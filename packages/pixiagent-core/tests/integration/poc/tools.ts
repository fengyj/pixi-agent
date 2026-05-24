import { z } from 'zod';
import { defineTool, Tool } from '../../../src/tool';

const generateFutureWeather = () => {
  const today = new Date();
  const days = ['Sunny', 'Cloudy', 'Rainy', 'Windy', 'Partly cloudy'];

  return Array.from({ length: 3 }, (_, idx) => {
    const date = new Date(today);
    date.setDate(date.getDate() + idx);
    return {
      date: date.toISOString().slice(0, 10),
      high: 18 + idx * 2 + Math.floor(Math.random() * 4),
      low: 10 + idx * 2 + Math.floor(Math.random() * 3),
      condition: days[(today.getDate() + idx) % days.length],
    };
  });
};

const hashString = (value: string) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
};

const normalizePrice = (value: number) => Math.round(value * 100) / 100;

const generateStockPrices = (ticker: string, date: string) => {
  const base = 100 + (hashString(ticker) % 100);
  const daySeed = hashString(`${ticker}:${date}`) % 50;
  const open = normalizePrice(base + daySeed * 0.3);
  const close = normalizePrice(open + ((hashString(`${ticker}:${date}:c`) % 200) - 100) * 0.05);
  const high = normalizePrice(Math.max(open, close) + ((hashString(`${ticker}:${date}:h`) % 80) * 0.1));
  const low = normalizePrice(Math.min(open, close) - ((hashString(`${ticker}:${date}:l`) % 80) * 0.1));
  const volume = 1_000_000 + (hashString(`${ticker}:${date}:v`) % 5_000_000);

  return {
    ticker: ticker.toUpperCase(),
    date,
    open,
    high,
    low,
    close,
    volume,
  };
};

export const fakeWeatherTool = defineTool({
  name: 'future_weather',
  description: '返回从今天开始未来 3 天的天气情况，包括日期、最高温度、最低温度和天气描述。',
  schema: z.object({}),
  handler: async () => ({
    forecasts: generateFutureWeather(),
  }),
  funcChecker: () => true,
});

const stockQueryParameterSchema = z.object({
  ticker: z.string().min(1),
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, '日期必须为 YYYY-MM-DD 格式'),
});
type StockQueryParameter = z.infer<typeof stockQueryParameterSchema>;

export const fakeStockPriceTool = defineTool({
  name: 'stock_ohlc',
  description: '根据输入的日期和股票代码返回该日期对应的 OHLC 股价数据。',
  schema: stockQueryParameterSchema,
  handler: async (input: StockQueryParameter) => {
    const { ticker, date } = input;
    const parsed = new Date(date);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error('无效日期，请使用 YYYY-MM-DD 格式');
    }

    return generateStockPrices(ticker, date);
  },
  funcChecker: () => true,
});

export const fakeToolset = {
  name: 'poc',
  description: '测试用的假工具集',
  tools: [fakeWeatherTool, fakeStockPriceTool],
};

export const executeToolCall = async (
    toolName: string, 
    input: unknown, 
    tools: Tool[]) => {
  const tool = tools.find(t => t.definition.name === toolName);
  if (!tool) {
    throw new Error(`工具 ${toolName} 未找到`);
  }

  if (!tool.funcChecker()) {
    throw new Error(`工具 ${toolName} 当前不可用`);
  }

  const parsedInput = tool.schema.parse(input);
  return await tool.handler(parsedInput);
}
