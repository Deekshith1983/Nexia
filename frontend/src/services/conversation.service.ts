import { request } from "./api"
import type {
  ConversationCreate,
  ConversationRead,
  ConversationUpdate,
  ConversationWithHistory,
} from "../types/conversation"
import type { ChatMessage } from "../types/message"

const API_V1 = "/api/v1"

export const ConversationService = {
  async listConversations(limit: number = 50, offset: number = 0): Promise<ConversationRead[]> {
    return request<ConversationRead[]>(`${API_V1}/conversations?limit=${limit}&offset=${offset}`)
  },

  async getConversation(id: string, messageLimit: number = 50): Promise<ConversationWithHistory> {
    return request<ConversationWithHistory>(`${API_V1}/conversations/${id}?message_limit=${messageLimit}`)
  },

  async createConversation(data: ConversationCreate = {}): Promise<ConversationRead> {
    return request<ConversationRead>(`${API_V1}/conversations`, {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  async updateConversation(id: string, data: ConversationUpdate): Promise<ConversationRead> {
    return request<ConversationRead>(`${API_V1}/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    })
  },

  async deleteConversation(id: string): Promise<void> {
    return request<void>(`${API_V1}/conversations/${id}`, {
      method: "DELETE",
    })
  },

  async getMessages(
    conversationId: string,
    limit: number = 50,
    signal?: AbortSignal
  ): Promise<ChatMessage[]> {
    return request<ChatMessage[]>(
      `${API_V1}/conversations/${conversationId}/messages?limit=${limit}`,
      { signal }
    )
  },
}

