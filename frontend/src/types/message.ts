export type MessageRole = "user" | "assistant" | "system"

export type MessageStatus = "COMPLETED" | "IN_PROGRESS" | "CANCELLED" | "FAILED"

export interface SourceReference {
  chunk_id: number
  document_title: string
  source_url: string
  section_title?: string
  passage_text: string
  rrf_score?: number
  retrieval_method?: string
}

export interface ChatMessage {
  id: string
  conversation_id: string
  role: MessageRole
  content: string
  token_count?: number
  message_status: MessageStatus
  source_references?: SourceReference[]
  diagnostics?: Record<string, any>
  created_at: string
}

