import React, { useEffect } from "react"
import { useChatStore } from "../../stores/chat.store"
import { Sidebar } from "./Sidebar"
import { PanelLeft, Plus, Sparkles } from "lucide-react"

interface AppLayoutProps {
  children: React.ReactNode
  onNewChat: () => void
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children, onNewChat }) => {
  const isSidebarOpen = useChatStore((s) => s.isSidebarOpen)
  const toggleSidebar = useChatStore((s) => s.toggleSidebar)
  const setSidebarOpen = useChatStore((s) => s.setSidebarOpen)

  // Handle responsive initial sidebar state and window resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false)
      } else {
        setSidebarOpen(true)
      }
    }

    handleResize()
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [setSidebarOpen])

  // Global keyboard shortcuts: Cmd/Ctrl + N for New Chat, Cmd/Ctrl + B for Sidebar toggle
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
        e.preventDefault()
        onNewChat()
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "b") {
        e.preventDefault()
        toggleSidebar()
      }
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [onNewChat, toggleSidebar])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <Sidebar onNewChat={onNewChat} />

      {/* Main Workspace */}
      <div className="flex flex-col flex-1 h-full min-w-0 overflow-hidden relative">
        {/* Top Navbar / Header */}
        <header className="flex items-center justify-between h-14 px-4 border-b border-border/40 bg-background/80 backdrop-blur-md z-10 select-none">
          <div className="flex items-center gap-3">
            {!isSidebarOpen && (
              <button
                onClick={toggleSidebar}
                type="button"
                title="Open sidebar (Ctrl+B)"
                className="p-2 rounded-xl text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              >
                <PanelLeft className="w-4 h-4" />
              </button>
            )}

            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight">Nexia</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onNewChat}
              type="button"
              title="New Chat (Ctrl+N)"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-muted hover:bg-muted/80 text-foreground transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">New Chat</span>
            </button>
          </div>
        </header>

        {/* Chat / Content Area */}
        <main className="flex flex-col flex-1 h-[calc(100vh-3.5rem)] overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  )
}

