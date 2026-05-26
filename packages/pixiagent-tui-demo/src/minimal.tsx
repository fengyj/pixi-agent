import React, { useState } from 'react';
import { Box, Text, useApp, useInput, render } from 'ink';

const Minimal = (): JSX.Element => {
  const { exit } = useApp();
  const [displayText, setDisplayText] = useState('Minimal TUI — type a message and press Ctrl+Enter to echo it.');
  const [inputText, setInputText] = useState('');

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
      return;
    }

    if ((key.ctrl || key.meta) && (key.return || key.enter)) {
      const trimmed = inputText.trim();
      if (trimmed === '/exit') {
        exit();
        return;
      }
      if (trimmed.length > 0) {
        setDisplayText(`Echo: ${trimmed}`);
      }
      setInputText('');
      return;
    }

    if (key.return || key.enter) {
      setInputText((prev) => `${prev}\n`);
      return;
    }

    if (key.backspace) {
      setInputText((prev) => prev.slice(0, -1));
      return;
    }

    if (!key.ctrl && !key.meta && input) {
      setInputText((prev) => prev + input);
    }
  });

  const inputLines = inputText.length > 0 ? inputText.split('\n') : [''];

  return (
    <Box flexDirection="column" padding={1} borderStyle="round" borderColor="#3a4a5a">
      <Text>{displayText}</Text>
      <Box flexDirection="column" marginTop={1} padding={1} borderStyle="single" borderColor="#4b5563" minHeight={4}>
        {inputLines.map((line, index) => (
          <Text key={index}>{line || ' '}</Text>
        ))}
      </Box>
      <Text color="#94a3b8">Ctrl+Enter to send, Enter for newline, Ctrl+C to quit.</Text>
    </Box>
  );
};

render(<Minimal />);
