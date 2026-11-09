import React from "react"
import { useChatStore } from "../../stores/chat.store"
import { SidebarHeader } from "./SidebarHeader"
import { ConversationList } from "./ConversationList"

interface SidebarProps {
  onNewChat: () => void
}

export const Sidebar: React.FC<SidebarProps> = ({ onNewChat }) => {
  const isSidebarOpen = useChatStore((s) => s.isSidebarOpen)
  const setSidebarOpen = useChatStore((s) => s.setSidebarOpen)

  const handleNewChat = () => {
    onNewChat()
    // On mobile screens, auto-close the drawer on selection
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-xs md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col w-[260px] sm:w-[280px] bg-sidebar border-r border-sidebar-border transition-transform duration-300 ease-in-out ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:-translate-x-full md:w-0 md:border-r-0"
        } overflow-hidden`}
      >
        <SidebarHeader onNewChat={handleNewChat} onCloseMobile={() => setSidebarOpen(false)} />
        <ConversationList />
      </aside>
    </>
  )
}

