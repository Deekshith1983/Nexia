import { useQuery } from "@tanstack/react-query"
import { ConversationService } from "../services/conversation.service"
import type { ChatMessage } from "../types/message"

export const messagesQueryKey = (conversationId: string | null) => ["messages", conversationId]

export function useConversationMessages(conversationId: string | null) {
  const query = useQuery<ChatMessage[]>({
    queryKey: messagesQueryKey(conversationId),
    queryFn: async ({ signal }) => {
      if (!conversationId) return []
      return ConversationService.getMessages(conversationId, 100, signal)
    },
    enabled: !!conversationId,
    staleTime: 60_000,   // 1 minute fresh
    gcTime: 300_000,     // 5 minutes in cache
  })

  return {
    messages: query.data || [],
    isLoading: query.isLoading && !!conversationId,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  }
}
