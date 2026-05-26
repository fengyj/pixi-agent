/**
 * Hook for managing scrollable content within a fixed-height viewport.
 *
 * Tracks scroll offset and provides navigation actions (page up/down,
 * line-level scroll, jump to top/bottom).
 */

import { useCallback, useMemo, useState } from 'react';
import { clamp } from '../utils/format';

export type ScrollActions = {
  /** Scroll up by half a page. */
  pageUp: () => void;
  /** Scroll down by half a page. */
  pageDown: () => void;
  /** Scroll up by one line. */
  lineUp: () => void;
  /** Scroll down by one line. */
  lineDown: () => void;
  /** Jump to the oldest (topmost) content. */
  scrollToTop: () => void;
  /** Jump to the latest (bottommost) content. */
  scrollToBottom: () => void;
  /** Reset scroll to bottom (e.g., when new content arrives). */
  resetScroll: () => void;
};

export type ScrollableViewState = {
  /** Lines visible within the current viewport. */
  visibleLines: string[];
  /** Human-readable scroll position label. */
  scrollPositionText: string;
  /** Scroll navigation actions. */
  actions: ScrollActions;
};

export function useScrollableView(
  allLines: string[],
  viewportHeight: number,
): ScrollableViewState {
  const [scrollOffset, setScrollOffset] = useState(0);

  const maxOffset = Math.max(0, allLines.length - viewportHeight);
  const offset = clamp(scrollOffset, 0, maxOffset);

  const visibleLines = useMemo(() => {
    const start = Math.max(0, allLines.length - viewportHeight - offset);
    const end = allLines.length - offset;
    return allLines.slice(start, end);
  }, [allLines, viewportHeight, offset]);

  const scrollPositionText = useMemo(() => {
    if (offset === 0) return 'bottom';
    const firstVisible = Math.max(1, allLines.length - viewportHeight - offset + 1);
    const lastVisible = Math.max(0, allLines.length - offset);
    return `${firstVisible}-${lastVisible}/${allLines.length}`;
  }, [allLines.length, viewportHeight, offset]);

  const halfPage = Math.max(3, Math.floor(viewportHeight / 2));

  const actions: ScrollActions = useMemo(
    () => ({
      pageUp: () =>
        setScrollOffset((cur) => clamp(cur + halfPage, 0, maxOffset)),
      pageDown: () =>
        setScrollOffset((cur) => clamp(cur - halfPage, 0, maxOffset)),
      lineUp: () =>
        setScrollOffset((cur) => clamp(cur + 1, 0, maxOffset)),
      lineDown: () =>
        setScrollOffset((cur) => clamp(cur - 1, 0, maxOffset)),
      scrollToTop: () => setScrollOffset(maxOffset),
      scrollToBottom: () => setScrollOffset(0),
      resetScroll: () => setScrollOffset(0),
    }),
    [halfPage, maxOffset],
  );

  return { visibleLines, scrollPositionText, actions };
}
