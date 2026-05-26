/**
 * Ink-compatible error boundary to catch render errors and prevent
 * the terminal from being left in a broken state (raw mode, hidden cursor).
 */

import React, { Component, type ReactNode } from 'react';
import { Text, Box } from 'ink';
import { COLORS } from '../constants';

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  override render(): ReactNode {
    if (this.state.error) {
      return (
        <Box flexDirection="column" padding={1} borderStyle="round" borderColor={COLORS.frame}>
          <Text color="red" bold>
            ⚠ Fatal Error
          </Text>
          <Text color={COLORS.muted}>
            {this.state.error.message}
          </Text>
          <Text color={COLORS.muted} dimColor>
            {this.state.error.stack ?? '(no stack trace)'}
          </Text>
          <Text color={COLORS.warning}>
            Press Ctrl+C to exit.
          </Text>
        </Box>
      );
    }

    return this.props.children;
  }
}
