import { describe, it, expect, beforeEach } from "vitest"
import { useChatStore } from "../stores/chat.store"

describe("ChatStore (Zustand)", () => {
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

  it("handles active conversation selection", () => {
    const store = useChatStore.getState()
    expect(store.activeConversationId).toBeNull()

    store.setActiveConversationId("conv-abc-123")
    expect(useChatStore.getState().activeConversationId).toBe("conv-abc-123")

    store.setActiveConversationId(null)
    expect(useChatStore.getState().activeConversationId).toBeNull()
  })

  it("toggles sidebar open state", () => {
    const store = useChatStore.getState()
    expect(store.isSidebarOpen).toBe(true)

    store.toggleSidebar()
    expect(useChatStore.getState().isSidebarOpen).toBe(false)

    store.setSidebarOpen(true)
    expect(useChatStore.getState().isSidebarOpen).toBe(true)
  })

  it("manages streaming lifecycle", () => {
    const store = useChatStore.getState()
    expect(store.isGenerating).toBe(false)

    store.startGeneration("req-1", "conv-1")
    const generatingState = useChatStore.getState()
    expect(generatingState.isGenerating).toBe(true)
    expect(generatingState.activeRequestId).toBe("req-1")
    expect(generatingState.activeStreamingConversationId).toBe("conv-1")

    store.appendStreamToken("Chunk 1 ")
    store.appendStreamToken("Chunk 2")
    expect(useChatStore.getState().streamingAnswer).toBe("Chunk 1 Chunk 2")

    store.stopGeneration()
    const stoppedState = useChatStore.getState()
    expect(stoppedState.isGenerating).toBe(false)
    expect(stoppedState.activeRequestId).toBeNull()
    expect(stoppedState.activeStreamingConversationId).toBeNull()
    expect(stoppedState.streamingAnswer).toBe("")
  })

  it("toggles theme correctly", () => {
    const store = useChatStore.getState()
    expect(store.theme).toBe("dark")

    store.toggleTheme()
    expect(useChatStore.getState().theme).toBe("light")

    store.setTheme("dark")
    expect(useChatStore.getState().theme).toBe("dark")
  })
})

