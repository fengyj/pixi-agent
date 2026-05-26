/**
 * Text input area with a blinking cursor effect.
 */

import React, { useEffect, useState } from 'react';
import { Box, Text } from 'ink';
import { COLORS, MIN_INPUT_HEIGHT } from '../constants';

type InputPanelProps = {
  /** Current input text (may contain newlines). */
  text: string;
};

const CURSOR_BLINK_MS = 530;

export const InputPanel = ({ text }: InputPanelProps): JSX.Element => {
  const [cursorVisible, setCursorVisible] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setCursorVisible((current) => !current);
    }, CURSOR_BLINK_MS);
    return () => clearInterval(interval);
  }, []);

  const cursorChar = cursorVisible ? '▉' : ' ';
  const isPlaceholder = text.length === 0;
  const placeholderText = 'Type your message... (Ctrl+S to send)';
  const rawLines = isPlaceholder ? [placeholderText] : text.split('\n');
  const lastLineIndex = rawLines.length - 1;

  return (
    <Box
      flexDirection="column"
      borderStyle="single"
      borderColor={COLORS.border}
      paddingLeft={1}
      paddingRight={1}
      paddingTop={0}
      paddingBottom={0}
      minHeight={MIN_INPUT_HEIGHT}
      marginTop={0}
    >
      {rawLines.map((line, index) => {
        if (isPlaceholder) {
          return (
            <Text key={`input-ph-${index}`} color={COLORS.placeholder}>
              {cursorChar}
              {line}
            </Text>
          );
        }

        return (
          <Text key={`input-${index}`}>
            {line}
            {index === lastLineIndex ? cursorChar : ''}
          </Text>
        );
      })}
    </Box>
  );
};
