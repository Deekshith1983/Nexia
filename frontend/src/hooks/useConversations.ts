import { useMemo } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ConversationService } from "../services/conversation.service"
import type { ConversationCreate, ConversationRead, ConversationUpdate } from "../types/conversation"
import { useChatStore } from "../stores/chat.store"

export const CONVERSATIONS_QUERY_KEY = ["conversations"]

export function useConversations() {
  const queryClient = useQueryClient()
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const setActiveConversationId = useChatStore((s) => s.setActiveConversationId)

  const conversationsQuery = useQuery({
    queryKey: CONVERSATIONS_QUERY_KEY,
    queryFn: () => ConversationService.listConversations(),
    staleTime: 30_000, // 30 seconds fresh
    gcTime: 300_000,    // 5 minutes garbage collection
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
      queryClient.setQueryData<ConversationRead[]>(CONVERSATIONS_QUERY_KEY, (prev = []) => {
        return [newConv, ...prev.filter((c) => c.id !== newConv.id)]
      })
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
      setActiveConversationId(newConv.id)
    },
  })

  // Optimistic Rename Mutation with rollback
  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: ConversationUpdate }) =>
      ConversationService.updateConversation(id, data),
    onMutate: async ({ id, data }) => {
      await queryClient.cancelQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
      const previousConversations = queryClient.getQueryData<ConversationRead[]>(CONVERSATIONS_QUERY_KEY)

      if (previousConversations) {
        queryClient.setQueryData<ConversationRead[]>(
          CONVERSATIONS_QUERY_KEY,
          previousConversations.map((conv) =>
            conv.id === id ? { ...conv, ...data, updated_at: new Date().toISOString() } : conv
          )
        )
      }

      return { previousConversations }
    },
    onError: (_err, _variables, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(CONVERSATIONS_QUERY_KEY, context.previousConversations)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
    },
  })

  // Optimistic Delete Mutation with rollback
  const deleteMutation = useMutation({
    mutationFn: (id: string) => ConversationService.deleteConversation(id),
    onMutate: async (deletedId) => {
      await queryClient.cancelQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
      const previousConversations = queryClient.getQueryData<ConversationRead[]>(CONVERSATIONS_QUERY_KEY)

      if (previousConversations) {
        queryClient.setQueryData<ConversationRead[]>(
          CONVERSATIONS_QUERY_KEY,
          previousConversations.filter((conv) => conv.id !== deletedId)
        )
      }

      return { previousConversations }
    },
    onError: (_err, _deletedId, context) => {
      if (context?.previousConversations) {
        queryClient.setQueryData(CONVERSATIONS_QUERY_KEY, context.previousConversations)
      }
    },
    onSuccess: (_, deletedId) => {
      if (activeConversationId === deletedId) {
        setActiveConversationId(null)
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
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
