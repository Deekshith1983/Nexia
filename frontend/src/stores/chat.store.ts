import { create } from "zustand"

interface ChatState {
  // Navigation & Conversation selection
  activeConversationId: string | null
  setActiveConversationId: (id: string | null) => void

  // Sidebar controls
  isSidebarOpen: boolean
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void

  // Streaming & Generation state
  isGenerating: boolean
  activeRequestId: string | null
  activeStreamingConversationId: string | null
  streamingAnswer: string
  streamingStage: string | null
  startGeneration: (requestId: string, conversationId: string) => void
  appendStreamToken: (token: string) => void
  appendStreamBatch: (batch: string) => void
  setStreamingAnswer: (answer: string) => void
  setStreamStage: (stage: string | null) => void
  stopGeneration: () => void

  // UI Theme
  theme: "dark" | "light"
  toggleTheme: () => void
  setTheme: (theme: "dark" | "light") => void
}

const getInitialTheme = (): "dark" | "light" => {
  if (typeof window !== "undefined") {
    const stored = localStorage.getItem("nexia-theme")
    if (stored === "light" || stored === "dark") return stored
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) return "dark"
  }
  return "dark"
}

export const useChatStore = create<ChatState>((set) => ({
  activeConversationId: null,
  setActiveConversationId: (id) =>
    set({
      activeConversationId: id,
    }),

  isSidebarOpen: true,
  toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  setSidebarOpen: (open) => set({ isSidebarOpen: open }),

  isGenerating: false,
  activeRequestId: null,
  activeStreamingConversationId: null,
  streamingAnswer: "",
  streamingStage: null,

  startGeneration: (requestId, conversationId) =>
    set({
      isGenerating: true,
      activeRequestId: requestId,
      activeStreamingConversationId: conversationId,
      streamingAnswer: "",
      streamingStage: "Gathering evidence...",
    }),

  appendStreamToken: (token) =>
    set((state) => ({
      streamingAnswer: state.streamingAnswer + token,
    })),

  appendStreamBatch: (batch) =>
    set((state) => ({
      streamingAnswer: state.streamingAnswer + batch,
    })),

  setStreamingAnswer: (answer) => set({ streamingAnswer: answer }),

  setStreamStage: (stage) => set({ streamingStage: stage }),

  stopGeneration: () =>
    set({
      isGenerating: false,
      activeRequestId: null,
      activeStreamingConversationId: null,
      streamingAnswer: "",
      streamingStage: null,
    }),

  theme: getInitialTheme(),
  toggleTheme: () =>
    set((state) => {
      const next = state.theme === "dark" ? "light" : "dark"
      if (typeof window !== "undefined") {
        localStorage.setItem("nexia-theme", next)
        document.documentElement.classList.toggle("dark", next === "dark")
      }
      return { theme: next }
    }),
  setTheme: (theme) => {
    if (typeof window !== "undefined") {
      localStorage.setItem("nexia-theme", theme)
      document.documentElement.classList.toggle("dark", theme === "dark")
    }
    set({ theme })
  },
}))

