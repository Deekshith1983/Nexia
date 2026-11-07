import React from "react"
import { Plus, Sparkles, Moon, Sun, PanelLeftClose } from "lucide-react"
import { useChatStore } from "../../stores/chat.store"

interface SidebarHeaderProps {
  onNewChat: () => void
  onCloseMobile?: () => void
}

export const SidebarHeader: React.FC<SidebarHeaderProps> = ({ onNewChat, onCloseMobile }) => {
  const { toggleSidebar, theme, toggleTheme } = useChatStore()

  return (
    <div className="flex flex-col gap-3 p-3 border-b border-sidebar-border">
      {/* Brand & Toggles */}
      <div className="flex items-center justify-between px-2">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h1 className="font-semibold text-base tracking-tight text-foreground">Nexia</h1>
            <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
              Wikipedia Grounded
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            type="button"
            title={theme === "dark" ? "Switch to Light mode" : "Switch to Dark mode"}
            className="p-1.5 rounded-lg text-sidebar-foreground hover:bg-sidebar-hover transition-colors"
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Collapse sidebar on desktop */}
          <button
            onClick={() => {
              if (onCloseMobile) onCloseMobile()
              else toggleSidebar()
            }}
            type="button"
            title="Collapse sidebar"
            className="p-1.5 rounded-lg text-sidebar-foreground hover:bg-sidebar-hover transition-colors"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* New Chat Button */}
      <button
        onClick={onNewChat}
        type="button"
        className="flex items-center justify-between w-full px-3.5 py-2.5 rounded-xl border border-sidebar-border bg-sidebar-hover/60 hover:bg-sidebar-hover text-foreground font-medium text-sm transition-all shadow-sm group"
      >
        <span className="flex items-center gap-2.5">
          <Plus className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors" />
          <span>New Chat</span>
        </span>
        <span className="text-[11px] text-muted-foreground font-mono opacity-60 group-hover:opacity-100">
          ⌘N
        </span>
      </button>
    </div>
  )
}

