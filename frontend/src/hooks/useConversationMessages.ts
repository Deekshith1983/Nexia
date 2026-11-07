import { useQuery } from "@tanstack/react-query"
import { ConversationService } from "../services/conversation.service"
import type { ChatMessage } from "../types/message"

export const messagesQueryKey = (conversationId: string | null) => ["messages", conversationId]

export function useConversationMessages(conversationId: string | null) {
  const query = useQuery<ChatMessage[]>({
    queryKey: messagesQueryKey(conversationId),
    queryFn: async () => {
      if (!conversationId) return []
      return ConversationService.getMessages(conversationId, 100)
    },
    enabled: !!conversationId,
    staleTime: 1000 * 60, // 1 minute
  })

  return {
    messages: query.data || [],
    isLoading: query.isLoading && !!conversationId,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  }
}

