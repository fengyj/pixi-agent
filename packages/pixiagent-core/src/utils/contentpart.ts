import { TextPart, ContentPart, ToolCallPart, ApiModes } from "../message";


/**
 * Normalize content into ContentPart[] so callers can process mixed string/parts
 * message content in a uniform way.
 */
function toContentParts(content?: string | Array<ContentPart>): Array<ContentPart> {
  if (content === undefined) {
    return [];
  }
  if (typeof content === 'string') {
    return [{ type: 'text', text: content } as TextPart];
  }
  return content;
}

/**
 * Concatenate two content fragments represented as text or parts.
 */
function concatContentParts(
  part1?: string | Array<ContentPart>,
  part2?: string | Array<ContentPart>,
): Array<ContentPart> {
  return [...toContentParts(part1), ...toContentParts(part2)];
}

function getContentDigest(
  content?: string | Array<ContentPart>,
): string | Array<ContentPart> | undefined {
  if (content === undefined) return undefined;

  const maxLength = 20;
  const headLength = 10;
  const tailLength = 5;

  const digestString = (value: string): string => {
    if (value.length <= maxLength) {
      return value;
    }
    return `${value.slice(0, headLength)}...${value.slice(-tailLength)}`;
  };

  const digestValue = (value: unknown): unknown => {
    if (typeof value === 'string') {
      return digestString(value);
    }

    if (Array.isArray(value)) {
      return value.map((item) => digestValue(item));
    }

    if (value instanceof Date) {
      return new Date(value.getTime());
    }

    if (value && typeof value === 'object') {
      const digestedObject: Record<string, unknown> = {};
      for (const [key, fieldValue] of Object.entries(value)) {
        digestedObject[key] = digestValue(fieldValue);
      }
      return digestedObject;
    }

    return value;
  };

  return digestValue(content) as string | Array<ContentPart>;
}

export const ContentParts = {
  toParts: toContentParts,
  concat: concatContentParts,
  digest: getContentDigest,
};