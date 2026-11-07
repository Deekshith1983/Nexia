export type ConversationStatus = "ACTIVE" | "ARCHIVED" | "DELETED"

export interface ConversationRead {
  id: string
  title: string
  status: ConversationStatus
  metadata: Record<string, any>
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationCreate {
  title?: string
  metadata?: Record<string, any>
}

export interface ConversationUpdate {
  title?: string
  status?: ConversationStatus
  metadata?: Record<string, any>
}

export interface ConversationWithHistory extends ConversationRead {
  messages: ChatMessage[]
}

import type { ChatMessage } from "./message"

