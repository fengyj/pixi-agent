/**
 * Chat message display area — renders visible lines within a scrollable viewport.
 */

import React from 'react';
import { Box, Text } from 'ink';
import { COLORS, MIN_MESSAGE_PANEL_HEIGHT } from '../constants';

type MessagePanelProps = {
  /** Lines of rendered text visible in the current viewport. */
  visibleLines: string[];
  /** Panel height in terminal rows. */
  height: number;
};

export const MessagePanel = ({
  visibleLines,
  height,
}: MessagePanelProps): JSX.Element => {
  return (
    <Box
      flexDirection="column"
      borderStyle="single"
      borderColor={COLORS.border}
      paddingLeft={1}
      paddingRight={1}
      paddingTop={0}
      paddingBottom={0}
      flexGrow={1}
      minHeight={Math.max(MIN_MESSAGE_PANEL_HEIGHT, height)}
    >
      {visibleLines.map((line, index) => (
        <Text key={`msg-${index}`} wrap="wrap">
          {line}
        </Text>
      ))}
    </Box>
  );
};
