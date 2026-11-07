import { useMemo } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ConversationService } from "../services/conversation.service"
import type { ConversationCreate, ConversationRead, ConversationUpdate } from "../types/conversation"
import { useChatStore } from "../stores/chat.store"

export const CONVERSATIONS_QUERY_KEY = ["conversations"]

export function useConversations() {
  const queryClient = useQueryClient()
  const { activeConversationId, setActiveConversationId } = useChatStore()

  const conversationsQuery = useQuery({
    queryKey: CONVERSATIONS_QUERY_KEY,
    queryFn: () => ConversationService.listConversations(),
    staleTime: 0,
  })

  // Defensive deduplication by unique conversation ID and ordering by updated_at desc
  const conversations = useMemo(() => {
    const raw = conversationsQuery.data || []
    const map = new Map<string, ConversationRead>()
    for (const c of raw) {
      if (c && c.id && !map.has(c.id)) {
        map.set(c.id, c)
      }
    }
    return Array.from(map.values()).sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    )
  }, [conversationsQuery.data])

  const createMutation = useMutation({
    mutationFn: (data?: ConversationCreate) => ConversationService.createConversation(data),
    onSuccess: (newConv) => {
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
      setActiveConversationId(newConv.id)
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: ConversationUpdate }) =>
      ConversationService.updateConversation(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => ConversationService.deleteConversation(id),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
      if (activeConversationId === deletedId) {
        setActiveConversationId(null)
      }
    },
  })

  return {
    conversations,
    isLoading: conversationsQuery.isLoading,
    isError: conversationsQuery.isError,
    error: conversationsQuery.error,
    refetch: conversationsQuery.refetch,
    createConversation: createMutation.mutateAsync,
    isCreating: createMutation.isPending,
    updateConversation: updateMutation.mutateAsync,
    isUpdating: updateMutation.isPending,
    deleteConversation: deleteMutation.mutateAsync,
    isDeleting: deleteMutation.isPending,
  }
}
