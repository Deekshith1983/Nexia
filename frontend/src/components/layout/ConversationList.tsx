import React, { useState } from "react"
import { useConversations } from "../../hooks/useConversations"
import { useChatStore } from "../../stores/chat.store"
import { ConversationItem } from "./ConversationItem"
import { ConfirmModal } from "../ui/ConfirmModal"
import type { ConversationRead } from "../../types/conversation"

export const ConversationList: React.FC = () => {
  const { conversations, isLoading, updateConversation, deleteConversation } = useConversations()
  const { activeConversationId, setActiveConversationId, isGenerating } = useChatStore()

  const [conversationToDelete, setConversationToDelete] = useState<ConversationRead | null>(null)

  const handleSelect = (id: string) => {
    if (isGenerating) return
    setActiveConversationId(id)
  }

  const handleRename = async (id: string, newTitle: string) => {
    try {
      await updateConversation({ id, data: { title: newTitle } })
    } catch (err) {
      console.error("Rename failed:", err)
    }
  }

  const handleConfirmDelete = async () => {
    if (conversationToDelete) {
      try {
        await deleteConversation(conversationToDelete.id)
      } catch (err) {
        console.error("Delete failed:", err)
      } finally {
        setConversationToDelete(null)
      }
    }
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2 p-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-9 w-full bg-sidebar-hover/40 rounded-xl animate-pulse" />
        ))}
      </div>
    )
  }

  if (!conversations || conversations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-6 text-center text-muted-foreground">
        <p className="text-xs">No conversations yet.</p>
        <p className="text-[11px] mt-1 text-muted-foreground/70">
          Start a new chat to explore knowledge.
        </p>
      </div>
    )
  }

  return (
    <>
      <div className="flex flex-col gap-1 p-2 overflow-y-auto flex-1">
        <div className="px-3 py-1.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
          Recent
        </div>
        {conversations.map((conv) => (
          <ConversationItem
            key={conv.id}
            conversation={conv}
            isActive={activeConversationId === conv.id}
            onSelect={() => handleSelect(conv.id)}
            onRename={(newTitle) => handleRename(conv.id, newTitle)}
            onDelete={() => setConversationToDelete(conv)}
          />
        ))}
      </div>

      <ConfirmModal
        isOpen={!!conversationToDelete}
        title="Delete conversation?"
        message={`Are you sure you want to delete "${conversationToDelete?.title || "this conversation"}"? This will permanently delete all its messages and history.`}
        confirmText="Delete"
        isDestructive
        onConfirm={handleConfirmDelete}
        onCancel={() => setConversationToDelete(null)}
      />
    </>
  )
}

