/**
 * Status bar showing agent status, scroll position, and token usage.
 */

import React from 'react';
import { Box, Text } from 'ink';
import type { UsageSnapshot } from '../types';
import { COLORS } from '../constants';
import { formatUsageTokens } from '../utils/format';

type StatusBarProps = {
  /** Current agent status label (e.g. "Ready", "Thinking..."). */
  statusLabel: string;
  /** Human-readable scroll position (e.g. "bottom", "1-20/45"). */
  scrollPositionText: string;
  /** Token usage for the last request. */
  lastRequestUsage: UsageSnapshot;
  /** Cumulative session token usage. */
  sessionUsage: UsageSnapshot;
};

export const StatusBar = ({
  statusLabel,
  scrollPositionText,
  lastRequestUsage,
  sessionUsage,
}: StatusBarProps): JSX.Element => {
  const statusText = `${statusLabel} | Pos: ${scrollPositionText}`;

  return (
    <>
      <Box justifyContent="flex-end" marginTop={1}>
        <Text color={COLORS.muted}>
          Total (I|O|C|T): {formatUsageTokens(sessionUsage)}
        </Text>
      </Box>
      <Box justifyContent="space-between">
        <Text color={COLORS.warning}>{statusText}</Text>
        <Text color={COLORS.warning}>
          Last  (I|O|C|T): {formatUsageTokens(lastRequestUsage)}
        </Text>
      </Box>
    </>
  );
};
