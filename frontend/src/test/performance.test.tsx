import { describe, it, expect, vi } from "vitest"
import { render, screen, fireEvent, renderHook, act } from "@testing-library/react"
import React from "react"
import "@testing-library/jest-dom/vitest"

import { Skeleton } from "../components/ui/Skeleton"
import { SidebarSkeleton } from "../components/layout/SidebarSkeleton"
import { MessageSkeleton } from "../components/chat/MessageSkeleton"
import { ErrorBoundary } from "../components/ui/ErrorBoundary"
import { throttle, rafThrottle } from "../utils/throttle"
import { debounce } from "../utils/debounce"
import { useVirtualizer } from "../hooks/useVirtualizer"

describe("Area 1: Skeleton Loaders", () => {
  it("renders basic Skeleton with pulse class", () => {
    const { container } = render(<Skeleton className="h-4 w-20" />)
    const el = container.firstChild as HTMLElement
    expect(el).toHaveClass("animate-pulse")
  })

  it("renders SidebarSkeleton with placeholder rows", () => {
    const { container } = render(<SidebarSkeleton />)
    const skeletons = container.querySelectorAll(".animate-pulse")
    expect(skeletons.length).toBeGreaterThanOrEqual(5)
  })

  it("renders MessageSkeleton with user and assistant placeholders", () => {
    const { container } = render(<MessageSkeleton />)
    const skeletons = container.querySelectorAll(".animate-pulse")
    expect(skeletons.length).toBeGreaterThan(6)
  })
})

describe("Area 9: Debouncing and Throttling Utilities", () => {
  it("throttle limits execution rate", () => {
    vi.useFakeTimers()
    const fn = vi.fn()
    const throttled = throttle(fn, 100)

    throttled("a")
    throttled("b")
    throttled("c")

    expect(fn).toHaveBeenCalledTimes(1)
    expect(fn).toHaveBeenCalledWith("a")

    vi.advanceTimersByTime(100)
    expect(fn).toHaveBeenCalledTimes(2)
    expect(fn).toHaveBeenLastCalledWith("c")

    vi.useRealTimers()
  })

  it("debounce delays invocation until delay period ends", () => {
    vi.useFakeTimers()
    const fn = vi.fn()
    const debounced = debounce(fn, 200)

    debounced(1)
    debounced(2)
    debounced(3)

    expect(fn).not.toHaveBeenCalled()

    vi.advanceTimersByTime(199)
    expect(fn).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1)
    expect(fn).toHaveBeenCalledTimes(1)
    expect(fn).toHaveBeenCalledWith(3)

    vi.useRealTimers()
  })
})

describe("Area 11: Error Boundary & Recovery", () => {
  const ProblemChild: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
    if (shouldThrow) {
      throw new Error("Simulated rendering failure")
    }
    return <div>Normal Content</div>
  }

  it("renders normal children when no error occurs", () => {
    render(
      <ErrorBoundary>
        <ProblemChild shouldThrow={false} />
      </ErrorBoundary>
    )
    expect(screen.getByText("Normal Content")).toBeInTheDocument()
  })

  it("catches error and displays fallback UI with retry button", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    render(
      <ErrorBoundary fallbackTitle="Custom error title">
        <ProblemChild shouldThrow={true} />
      </ErrorBoundary>
    )

    expect(screen.getByText("Custom error title")).toBeInTheDocument()
    expect(screen.getByText("Try again")).toBeInTheDocument()

    consoleSpy.mockRestore()
  })
})

describe("Area 2: Virtualizer Hook", () => {
  it("calculates initial virtual items based on estimateSize and overscan", () => {
    const mockContainer = document.createElement("div")
    Object.defineProperty(mockContainer, "clientHeight", { value: 400, configurable: true })
    Object.defineProperty(mockContainer, "scrollTop", { value: 0, configurable: true })

    const { result } = renderHook(() =>
      useVirtualizer({
        count: 50,
        getScrollElement: () => mockContainer,
        estimateSize: () => 100,
        overscan: 2,
      })
    )

    expect(result.current.getTotalSize()).toBe(5000)
    const items = result.current.getVirtualItems()
    expect(items.length).toBeGreaterThan(0)
    expect(items[0].index).toBe(0)
    expect(items[0].size).toBe(100)
  })
})

