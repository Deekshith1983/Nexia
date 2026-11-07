import React from "react"
import { AppLayout } from "../components/layout/AppLayout"
import { ChatWindow } from "../components/chat/ChatWindow"
import { useChatStore } from "../stores/chat.store"

export const ChatPage: React.FC = () => {
  const { setActiveConversationId, isGenerating } = useChatStore()

  const handleNewChat = () => {
    if (isGenerating) return
    setActiveConversationId(null)
  }

  return (
    <AppLayout onNewChat={handleNewChat}>
      <ChatWindow />
    </AppLayout>
  )
}

