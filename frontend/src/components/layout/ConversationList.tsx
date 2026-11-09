import React, { useState, useCallback } from "react"
import { useConversations } from "../../hooks/useConversations"
import { useChatStore } from "../../stores/chat.store"
import { ConversationItem } from "./ConversationItem"
import { ConfirmModal } from "../ui/ConfirmModal"
import { SidebarSkeleton } from "./SidebarSkeleton"
import type { ConversationRead } from "../../types/conversation"

export const ConversationList: React.FC = () => {
  const { conversations, isLoading, updateConversation, deleteConversation } = useConversations()
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const setActiveConversationId = useChatStore((s) => s.setActiveConversationId)
  const isGenerating = useChatStore((s) => s.isGenerating)

  const [conversationToDelete, setConversationToDelete] = useState<ConversationRead | null>(null)

  const handleSelect = useCallback(
    (id: string) => {
      if (isGenerating) return
      setActiveConversationId(id)
    },
    [isGenerating, setActiveConversationId]
  )

  const handleRename = useCallback(
    async (id: string, newTitle: string) => {
      try {
        await updateConversation({ id, data: { title: newTitle } })
      } catch (err) {
        console.error("Rename failed:", err)
      }
    },
    [updateConversation]
  )

  const handleConfirmDelete = useCallback(async () => {
    if (conversationToDelete) {
      try {
        await deleteConversation(conversationToDelete.id)
      } catch (err) {
        console.error("Delete failed:", err)
      } finally {
        setConversationToDelete(null)
      }
    }
  }, [conversationToDelete, deleteConversation])

  const handleDeleteTrigger = useCallback((conv: ConversationRead) => {
    setConversationToDelete(conv)
  }, [])

  if (isLoading) {
    return <SidebarSkeleton />
  }

  if (!conversations || conversations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-6 text-center text-muted-foreground animate-in fade-in duration-200">
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
            onDelete={() => handleDeleteTrigger(conv)}
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
