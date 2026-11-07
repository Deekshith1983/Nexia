import type { SourceReference } from "./message"

export interface AnswerRequest {
  question: string
  conversation_id?: string
  subject_entity?: string
  object_entity?: string
  predicate_filter?: string
  top_k_per_document?: number
  min_confidence?: number
  max_evidence_passages?: number
  force_refresh?: boolean
}

export interface AnswerResponse {
  question: string
  answer_text: string
  answer_status: "ANSWERED" | "INSUFFICIENT_EVIDENCE" | "CONFLICTING_EVIDENCE" | "LLM_FAILED"
  source_references: SourceReference[]
  conflicts: string[]
  sufficiency_rating: string
  retrieval_diagnostics?: Record<string, any>
  stage6_evidence_package?: Record<string, any>
  error_message?: string
  conversation_id?: string
  message_id?: string
}

export interface SSEStartEvent {
  request_id: string
  conversation_id?: string
  model: string
  status: string
}

export interface SSEProgressEvent {
  request_id: string
  stage: string
  message: string
}

export interface SSETokenEvent {
  request_id: string
  token: string
}

export interface SSECitationEvent {
  request_id: string
  citations: SourceReference[]
}

export interface SSECompleteEvent {
  request_id: string
  conversation_id?: string
  message_id?: string
  answer_text: string
  answer_status: string
  source_references: SourceReference[]
  diagnostics?: Record<string, any>
  timings?: Record<string, any>
}

export interface SSETitleEvent {
  request_id: string
  conversation_id?: string
  title: string
}

export interface SSEErrorEvent {
  request_id: string
  error_code: string
  message: string
}

export interface SSECancelledEvent {
  request_id: string
  reason: string
}

