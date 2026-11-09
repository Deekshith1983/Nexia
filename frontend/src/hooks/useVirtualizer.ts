import { useState, useEffect, useRef, useCallback, useMemo } from "react"

export interface VirtualItem {
  index: number
  start: number
  size: number
  end: number
}

export interface UseVirtualizerOptions {
  count: number
  getScrollElement: () => HTMLElement | null
  estimateSize: (index: number) => number
  overscan?: number
}

export interface VirtualizerInstance {
  getVirtualItems: () => VirtualItem[]
  getTotalSize: () => number
  measureElement: (element: HTMLElement | null) => void
  scrollToIndex: (index: number, options?: { align?: "start" | "center" | "end" | "auto" }) => void
  scrollToBottom: (smooth?: boolean) => void
  isScrolling: boolean
}

export function useVirtualizer({
  count,
  getScrollElement,
  estimateSize,
  overscan = 3,
}: UseVirtualizerOptions): VirtualizerInstance {
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(0)
  const [isScrolling, setIsScrolling] = useState(false)
  const isScrollingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Cache for dynamic heights: index -> measured height in px
  const measurementsRef = useRef<Map<number, number>>(new Map())
  const [, setMeasurementRevision] = useState(0)

  // ResizeObserver for measuring DOM elements dynamically
  const observerRef = useRef<ResizeObserver | null>(null)

  useEffect(() => {
    if (typeof ResizeObserver === "undefined") return

    observerRef.current = new ResizeObserver((entries) => {
      let changed = false
      for (const entry of entries) {
        const el = entry.target as HTMLElement
        const indexStr = el.dataset.index
        if (indexStr !== undefined) {
          const index = parseInt(indexStr, 10)
          const measuredHeight = Math.ceil(entry.borderBoxSize?.[0]?.blockSize ?? el.getBoundingClientRect().height)
          if (measuredHeight > 0 && measurementsRef.current.get(index) !== measuredHeight) {
            measurementsRef.current.set(index, measuredHeight)
            changed = true
          }
        }
      }
      if (changed) {
        setMeasurementRevision((v) => v + 1)
      }
    })

    return () => {
      observerRef.current?.disconnect()
    }
  }, [])

  // Listen to scroll container scroll and resize events
  useEffect(() => {
    const scrollEl = getScrollElement()
    if (!scrollEl) return

    setViewportHeight(scrollEl.clientHeight)
    setScrollTop(scrollEl.scrollTop)

    let rafId: number | null = null

    const handleScroll = () => {
      if (rafId !== null) return
      rafId = requestAnimationFrame(() => {
        rafId = null
        if (scrollEl) {
          setScrollTop(scrollEl.scrollTop)
          setIsScrolling(true)
          if (isScrollingTimeoutRef.current) {
            clearTimeout(isScrollingTimeoutRef.current)
          }
          isScrollingTimeoutRef.current = setTimeout(() => {
            setIsScrolling(false)
          }, 150)
        }
      })
    }

    const handleResize = () => {
      if (scrollEl) {
        setViewportHeight(scrollEl.clientHeight)
        setScrollTop(scrollEl.scrollTop)
      }
    }

    scrollEl.addEventListener("scroll", handleScroll, { passive: true })
    window.addEventListener("resize", handleResize)

    let containerResizeObserver: ResizeObserver | null = null
    if (typeof ResizeObserver !== "undefined") {
      containerResizeObserver = new ResizeObserver(() => {
        if (scrollEl) {
          setViewportHeight(scrollEl.clientHeight)
        }
      })
      containerResizeObserver.observe(scrollEl)
    }

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId)
      if (isScrollingTimeoutRef.current) clearTimeout(isScrollingTimeoutRef.current)
      scrollEl.removeEventListener("scroll", handleScroll)
      window.removeEventListener("resize", handleResize)
      containerResizeObserver?.disconnect()
    }
  }, [getScrollElement])

  // Calculate item offsets and sizes based on measurements or estimates
  const { offsets, totalSize } = useMemo(() => {
    const itemOffsets = new Array<number>(count)
    let currentOffset = 0

    for (let i = 0; i < count; i++) {
      itemOffsets[i] = currentOffset
      const size = measurementsRef.current.get(i) ?? estimateSize(i)
      currentOffset += size
    }

    return {
      offsets: itemOffsets,
      totalSize: currentOffset,
    }
  }, [count, estimateSize])

  // Callback ref passed to each item's DOM container
  const measureElement = useCallback((element: HTMLElement | null) => {
    if (!element) return
    const indexStr = element.dataset.index
    if (indexStr === undefined) return
    const index = parseInt(indexStr, 10)

    // Immediate measurement
    const height = Math.ceil(element.getBoundingClientRect().height)
    if (height > 0 && measurementsRef.current.get(index) !== height) {
      measurementsRef.current.set(index, height)
      setMeasurementRevision((v) => v + 1)
    }

    // Connect to ResizeObserver for dynamic size shifts (e.g. Markdown rendering, image loading)
    observerRef.current?.observe(element)
  }, [])

  // Calculate visible range with overscan
  const getVirtualItems = useCallback((): VirtualItem[] => {
    if (count === 0 || viewportHeight === 0) return []

    const rangeStart = Math.max(0, scrollTop)
    const rangeEnd = scrollTop + viewportHeight

    // Find start index (binary search)
    let startIndex = 0
    let low = 0
    let high = count - 1
    while (low <= high) {
      const mid = Math.floor((low + high) / 2)
      const offset = offsets[mid]
      const size = measurementsRef.current.get(mid) ?? estimateSize(mid)
      if (offset + size < rangeStart) {
        low = mid + 1
      } else {
        startIndex = mid
        high = mid - 1
      }
    }

    // Find end index
    let endIndex = startIndex
    while (endIndex < count && offsets[endIndex] < rangeEnd) {
      endIndex++
    }

    // Apply overscan
    const overscanStart = Math.max(0, startIndex - overscan)
    const overscanEnd = Math.min(count - 1, endIndex + overscan)

    const virtualItems: VirtualItem[] = []
    for (let i = overscanStart; i <= overscanEnd; i++) {
      const size = measurementsRef.current.get(i) ?? estimateSize(i)
      virtualItems.push({
        index: i,
        start: offsets[i] ?? 0,
        size,
        end: (offsets[i] ?? 0) + size,
      })
    }

    return virtualItems
  }, [count, viewportHeight, scrollTop, offsets, estimateSize, overscan])

  const getTotalSize = useCallback(() => totalSize, [totalSize])

  const scrollToIndex = useCallback(
    (index: number, options?: { align?: "start" | "center" | "end" | "auto" }) => {
      const scrollEl = getScrollElement()
      if (!scrollEl || index < 0 || index >= count) return

      const offset = offsets[index] ?? 0
      const size = measurementsRef.current.get(index) ?? estimateSize(index)
      const align = options?.align ?? "auto"

      let targetScrollTop = offset
      if (align === "end") {
        targetScrollTop = offset + size - scrollEl.clientHeight
      } else if (align === "center") {
        targetScrollTop = offset + size / 2 - scrollEl.clientHeight / 2
      } else if (align === "auto") {
        if (offset < scrollEl.scrollTop) {
          targetScrollTop = offset
        } else if (offset + size > scrollEl.scrollTop + scrollEl.clientHeight) {
          targetScrollTop = offset + size - scrollEl.clientHeight
        } else {
          return // Already visible
        }
      }

      scrollEl.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: "smooth",
      })
    },
    [count, getScrollElement, offsets, estimateSize]
  )

  const scrollToBottom = useCallback(
    (smooth = true) => {
      const scrollEl = getScrollElement()
      if (!scrollEl) return
      scrollEl.scrollTo({
        top: scrollEl.scrollHeight,
        behavior: smooth ? "smooth" : "auto",
      })
    },
    [getScrollElement]
  )

  return {
    getVirtualItems,
    getTotalSize,
    measureElement,
    scrollToIndex,
    scrollToBottom,
    isScrolling,
  }
}
