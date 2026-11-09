/**
 * ==============================================================================
 * NEXIA CONSOLIDATED FRONTEND TEST SUITE (nexia.test.tsx)
 * ==============================================================================
 * Author: Deekshith1983 <dikideeku@gmail.com>
 * Project: Nexia — Intelligent Wikipedia-Grounded Conversational AI Assistant
 *
 * Consolidated frontend test suite covering:
 *   Section 1: UI Component Rendering & Interaction
 *   Section 2: Zustand Chat Store State Management
 *   Section 3: SSE Streaming Engine & "Thinking..." Indicator
 *   Section 4: Enforced Retrieval Configuration (3, 0.4, 9)
 *   Section 5: Performance Optimizations, Virtualizer & Error Boundaries
 *   Section 6: Single Wikipedia Article Citation & Link Deduplication
 * ==============================================================================
 */

import React from "react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import "@testing-library/jest-dom/vitest"
import { render, screen, fireEvent, renderHook, act } from "@testing-library/react"

// Components & UI
import { WelcomeScreen } from "../components/chat/WelcomeScreen"
import { MessageBubble } from "../components/chat/MessageBubble"
import { MessageComposer } from "../components/chat/MessageComposer"
import { StreamingIndicator } from "../components/chat/StreamingIndicator"
import { ConfirmModal } from "../components/ui/ConfirmModal"
import { Skeleton } from "../components/ui/Skeleton"
import { SidebarSkeleton } from "../components/layout/SidebarSkeleton"
import { MessageSkeleton } from "../components/chat/MessageSkeleton"
import { ErrorBoundary } from "../components/ui/ErrorBoundary"

// Stores & State
import { useChatStore } from "../stores/chat.store"

// Services & Config
import { RETRIEVAL_CONFIG, buildAnswerPayload } from "../config/retrieval.config"
import { AnswerService } from "../services/answer.service"
import { streamAnswer } from "../services/streaming.service"
import * as api from "../services/api"
import type { AnswerRequest } from "../types/api"

// Utilities & Hooks
import { throttle, rafThrottle } from "../utils/throttle"
import { debounce } from "../utils/debounce"
import { useVirtualizer } from "../hooks/useVirtualizer"


// ==============================================================================
// SECTION 1: UI Component Rendering & Interaction
// ==============================================================================

describe("Section 1: UI Component Rendering & Interaction", () => {
  it("renders WelcomeScreen with heading and clickable suggested prompts", () => {
    const onSelectPrompt = vi.fn()
    render(<WelcomeScreen onSelectPrompt={onSelectPrompt} />)

    expect(screen.getByText("How can I help you today?")).toBeInTheDocument()
    expect(
      screen.getByText(/Ask anything. Explore ideas, understand complex topics/i)
    ).toBeInTheDocument()

    const promptBtn = screen.getByText(/Who was Shivaji Maharaj/i)
    expect(promptBtn).toBeInTheDocument()

    fireEvent.click(promptBtn)
    expect(onSelectPrompt).toHaveBeenCalledWith(
      expect.stringContaining("Shivaji Maharaj")
    )
  })

  it("renders MessageBubble for user messages with correct sender and styling", () => {
    render(
      <MessageBubble
        message={{
          id: "msg-user-1",
          conversation_id: "conv-1",
          role: "user",
          content: "What is quantum mechanics?",
          message_status: "COMPLETED",
          created_at: new Date().toISOString(),
        }}
      />
    )

    expect(screen.getByText("You")).toBeInTheDocument()
    expect(screen.getByText("What is quantum mechanics?")).toBeInTheDocument()
  })

  it("renders MessageBubble for assistant answers with Markdown & single Wikipedia link", () => {
    render(
      <MessageBubble
        message={{
          id: "msg-assistant-1",
          conversation_id: "conv-1",
          role: "assistant",
          content:
            "Samudragupta was a Gupta ruler.\n\n[Allahabad Pillar Inscription](https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription)",
          message_status: "COMPLETED",
          created_at: new Date().toISOString(),
        }}
      />
    )

    expect(screen.getByText("Nexia")).toBeInTheDocument()
    expect(screen.getByText(/Samudragupta was a Gupta ruler/i)).toBeInTheDocument()

    const link = screen.getByRole("link", { name: "Allahabad Pillar Inscription" })
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute(
      "href",
      "https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription"
    )
  })

  it("handles MessageComposer text input, submission, and keyboard shortcuts", () => {
    const onSend = vi.fn()
    const onStop = vi.fn()

    const { rerender } = render(
      <MessageComposer onSend={onSend} onStop={onStop} isGenerating={false} />
    )

    const textarea = screen.getByPlaceholderText(/Ask Nexia anything/i)
    fireEvent.change(textarea, { target: { value: "Tell me about Ashoka the Great" } })
    expect(textarea).toHaveValue("Tell me about Ashoka the Great")

    const sendBtn = screen.getByLabelText("Send question")
    expect(sendBtn).not.toBeDisabled()

    fireEvent.click(sendBtn)
    expect(onSend).toHaveBeenCalledWith("Tell me about Ashoka the Great")

    // When generating:
    rerender(<MessageComposer onSend={onSend} onStop={onStop} isGenerating={true} />)
    const stopBtn = screen.getByLabelText("Stop generation")
    expect(stopBtn).toBeInTheDocument()
    fireEvent.click(stopBtn)
    expect(onStop).toHaveBeenCalled()
  })

  it("renders ConfirmModal and dispatches confirm/cancel actions", () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()

    render(
      <ConfirmModal
        isOpen={true}
        title="Delete Item?"
        message="Are you sure you want to delete this conversation?"
        onConfirm={onConfirm}
        onCancel={onCancel}
        isDestructive
      />
    )

    expect(screen.getByText("Delete Item?")).toBeInTheDocument()
    expect(screen.getByText("Are you sure you want to delete this conversation?")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Cancel"))
    expect(onCancel).toHaveBeenCalledTimes(1)

    fireEvent.click(screen.getByText("Confirm"))
    expect(onConfirm).toHaveBeenCalledTimes(1)
  })
})


// ==============================================================================
// SECTION 2: Zustand Chat Store State Management
// ==============================================================================

describe("Section 2: Zustand Chat Store State Management", () => {
  beforeEach(() => {
    useChatStore.setState({
      activeConversationId: null,
      isSidebarOpen: true,
      isGenerating: false,
      activeRequestId: null,
      activeStreamingConversationId: null,
      streamingAnswer: "",
      streamingStage: null,
      theme: "dark",
    })
  })

  it("handles active conversation selection and deselection", () => {
    const store = useChatStore.getState()
    expect(store.activeConversationId).toBeNull()

    store.setActiveConversationId("conv-abc-123")
    expect(useChatStore.getState().activeConversationId).toBe("conv-abc-123")

    store.setActiveConversationId(null)
    expect(useChatStore.getState().activeConversationId).toBeNull()
  })

  it("toggles and explicitly sets sidebar open/collapsed state", () => {
    const store = useChatStore.getState()
    expect(store.isSidebarOpen).toBe(true)

    store.toggleSidebar()
    expect(useChatStore.getState().isSidebarOpen).toBe(false)

    store.setSidebarOpen(true)
    expect(useChatStore.getState().isSidebarOpen).toBe(true)
  })

  it("manages streaming lifecycle from start to token appends to stop", () => {
    const store = useChatStore.getState()
    expect(store.isGenerating).toBe(false)

    store.startGeneration("req-100", "conv-200")
    const generatingState = useChatStore.getState()
    expect(generatingState.isGenerating).toBe(true)
    expect(generatingState.activeRequestId).toBe("req-100")
    expect(generatingState.activeStreamingConversationId).toBe("conv-200")

    store.appendStreamToken("Albert Einstein was ")
    store.appendStreamToken("a physicist.")
    expect(useChatStore.getState().streamingAnswer).toBe("Albert Einstein was a physicist.")

    store.stopGeneration()
    const stoppedState = useChatStore.getState()
    expect(stoppedState.isGenerating).toBe(false)
    expect(stoppedState.activeRequestId).toBeNull()
    expect(stoppedState.activeStreamingConversationId).toBeNull()
    expect(stoppedState.streamingAnswer).toBe("")
  })

  it("toggles and persists theme", () => {
    const store = useChatStore.getState()
    expect(store.theme).toBe("dark")

    store.toggleTheme()
    expect(useChatStore.getState().theme).toBe("light")

    store.setTheme("dark")
    expect(useChatStore.getState().theme).toBe("dark")
  })
})


// ==============================================================================
// SECTION 3: SSE Streaming Engine & "Thinking..." Indicator
// ==============================================================================

describe("Section 3: SSE Streaming Engine & 'Thinking...' Indicator", () => {
  it("renders StreamingIndicator with consistent 'Thinking...' text without technical jargon", () => {
    render(<StreamingIndicator stage="Thinking..." />)
    expect(screen.getByText("Thinking...")).toBeInTheDocument()
    expect(screen.queryByText(/gemma/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/knowledge base/i)).not.toBeInTheDocument()
  })

  it("parses sequential SSE events correctly (start, progress, token, complete, title)", async () => {
    const ssePayload = [
      'event: start\ndata: {"request_id":"req-123","model":"gemma3:4b","status":"STARTED"}\n\n',
      'event: progress\ndata: {"request_id":"req-123","stage":"retrieval","message":"Thinking..."}\n\n',
      'event: token\ndata: {"request_id":"req-123","token":"Hello "}\n\n',
      'event: token\ndata: {"request_id":"req-123","token":"world!"}\n\n',
      'event: complete\ndata: {"request_id":"req-123","answer_text":"Hello world!","answer_status":"ANSWERED","source_references":[]}\n\n',
      'event: title\ndata: {"request_id":"req-123","title":"Hello World Greetings"}\n\n',
    ].join("")

    const mockResponse = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload))
          controller.close()
        },
      }),
      { status: 200, headers: { "Content-Type": "text/event-stream" } }
    )

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockResponse)

    const onStart = vi.fn()
    const onProgress = vi.fn()
    const onToken = vi.fn()
    const onComplete = vi.fn()
    const onTitle = vi.fn()

    const request: AnswerRequest = { question: "Hi" }

    await streamAnswer(request, {
      onStart,
      onProgress,
      onToken,
      onComplete,
      onTitle,
    })

    expect(onStart).toHaveBeenCalledWith(
      expect.objectContaining({ request_id: "req-123", model: "gemma3:4b" })
    )
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ stage: "retrieval", message: "Thinking..." })
    )
    expect(onToken).toHaveBeenCalledTimes(2)
    expect(onToken).toHaveBeenNthCalledWith(1, "Hello ")
    expect(onToken).toHaveBeenNthCalledWith(2, "world!")
    expect(onComplete).toHaveBeenCalledWith(
      expect.objectContaining({ answer_text: "Hello world!", answer_status: "ANSWERED" })
    )
    expect(onTitle).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Hello World Greetings" })
    )
  })

  it("handles cancelled SSE events cleanly without crashing", async () => {
    const ssePayload = 'event: cancelled\ndata: {"request_id":"req-123","reason":"User cancelled"}\n\n'
    const mockResponse = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload))
          controller.close()
        },
      }),
      { status: 200 }
    )
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockResponse)

    const onCancelled = vi.fn()
    await streamAnswer({ question: "Cancel test" }, { onCancelled })

    expect(onCancelled).toHaveBeenCalledWith(
      expect.objectContaining({ request_id: "req-123", reason: "User cancelled" })
    )
  })
})


// ==============================================================================
// SECTION 4: Enforced Retrieval Configuration (3, 0.4, 9)
// ==============================================================================

describe("Section 4: Enforced Retrieval Configuration", () => {
  it("defines exact canonical retrieval configuration constants", () => {
    expect(RETRIEVAL_CONFIG).toEqual({
      top_k_per_document: 3,
      min_confidence: 0.4,
      max_evidence_passages: 9,
    })
  })

  it("automatically attaches enforced retrieval settings when omitted", () => {
    const payload = buildAnswerPayload({ question: "What is quantum computing?" })
    expect(payload).toEqual({
      question: "What is quantum computing?",
      top_k_per_document: 3,
      min_confidence: 0.4,
      max_evidence_passages: 9,
    })
  })

  it("overrides incorrect caller-supplied retrieval parameters", () => {
    const payload = buildAnswerPayload({
      question: "Explain relativity",
      top_k_per_document: 99 as any,
      min_confidence: 0.99 as any,
      max_evidence_passages: 1 as any,
    })

    expect(payload.top_k_per_document).toBe(3)
    expect(payload.min_confidence).toBe(0.4)
    expect(payload.max_evidence_passages).toBe(9)
  })

  it("normal and retry/regenerate payloads use the exact same configuration", () => {
    const normalPayload = buildAnswerPayload({ question: "Tell me about Ashoka" })
    const retryPayload = buildAnswerPayload({ question: "Tell me about Ashoka", top_k_per_document: 10 as any })

    expect(normalPayload.top_k_per_document).toBe(3)
    expect(retryPayload.top_k_per_document).toBe(3)
    expect(retryPayload.min_confidence).toBe(0.4)
    expect(retryPayload.max_evidence_passages).toBe(9)
  })
})


// ==============================================================================
// SECTION 5: Performance Optimizations, Virtualizer & Error Boundaries
// ==============================================================================

describe("Section 5: Performance Optimizations & Resilience", () => {
  it("renders Skeleton loaders with pulse animations", () => {
    const { container: sContainer } = render(<Skeleton className="h-4 w-20" />)
    expect(sContainer.firstChild).toHaveClass("animate-pulse")

    const { container: sbContainer } = render(<SidebarSkeleton />)
    expect(sbContainer.querySelectorAll(".animate-pulse").length).toBeGreaterThanOrEqual(4)

    const { container: mContainer } = render(<MessageSkeleton />)
    expect(mContainer.querySelectorAll(".animate-pulse").length).toBeGreaterThanOrEqual(4)
  })

  it("throttle limits execution rate within time window", () => {
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
    expect(fn).not.toHaveBeenCalled()

    vi.advanceTimersByTime(200)
    expect(fn).toHaveBeenCalledTimes(1)
    expect(fn).toHaveBeenCalledWith(2)

    vi.useRealTimers()
  })

  it("useVirtualizer computes correct total height and windowed slice", () => {
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

  it("ErrorBoundary catches runtime errors and presents fallback with retry", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    const ProblemChild: React.FC<{ shouldThrow: boolean }> = ({ shouldThrow }) => {
      if (shouldThrow) {
        throw new Error("Simulated render failure")
      }
      return <div>Normal Content</div>
    }

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


// ==============================================================================
// SECTION 6: Single Wikipedia Article Citation & Link Deduplication
// ==============================================================================

describe("Section 6: Single Wikipedia Article Citation & Link Deduplication", () => {
  it("renders exactly one Wikipedia article hyperlink at the very end of answer", () => {
    const content = (
      "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. " +
      "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect.\n\n" +
      "[Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)"
    )

    render(
      <MessageBubble
        message={{
          id: "einstein-msg-1",
          conversation_id: "conv-101",
          role: "assistant",
          content,
          message_status: "COMPLETED",
          created_at: new Date().toISOString(),
        }}
      />
    )

    const links = screen.getAllByRole("link", { name: "Albert Einstein" })
    expect(links).toHaveLength(1)
    expect(links[0]).toHaveAttribute("href", "https://en.wikipedia.org/wiki/Albert_Einstein")
    expect(screen.queryByText(/\[k\]/)).not.toBeInTheDocument()
    expect(screen.queryByText(/\[1\]/)).not.toBeInTheDocument()
  })
})
