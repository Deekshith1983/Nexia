import { useRef, useCallback, useEffect } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chat.store"
import { ConversationService } from "../services/conversation.service"
import { AnswerService } from "../services/answer.service"
import { streamAnswer } from "../services/streaming.service"
import { buildAnswerPayload } from "../config/retrieval.config"
import { messagesQueryKey } from "./useConversationMessages"
import { CONVERSATIONS_QUERY_KEY } from "./useConversations"
import type { ChatMessage } from "../types/message"
import type { ConversationRead } from "../types/conversation"

export function useChat() {
  const queryClient = useQueryClient()

  // Fine-grained selector subscriptions to avoid re-renders when other slices change
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const setActiveConversationId = useChatStore((s) => s.setActiveConversationId)
  const isGenerating = useChatStore((s) => s.isGenerating)
  const activeRequestId = useChatStore((s) => s.activeRequestId)
  const activeStreamingConversationId = useChatStore((s) => s.activeStreamingConversationId)
  const streamingAnswer = useChatStore((s) => s.streamingAnswer)
  const streamingStage = useChatStore((s) => s.streamingStage)
  const startGeneration = useChatStore((s) => s.startGeneration)
  const appendStreamBatch = useChatStore((s) => s.appendStreamBatch)
  const setStreamingAnswer = useChatStore((s) => s.setStreamingAnswer)
  const setStreamStage = useChatStore((s) => s.setStreamStage)
  const stopGeneration = useChatStore((s) => s.stopGeneration)

  const abortControllerRef = useRef<AbortController | null>(null)
  const tokenBufferRef = useRef<string>("")
  const rafIdRef = useRef<number | null>(null)

  // Flush any accumulated tokens in buffer to Zustand store
  const flushTokenBuffer = useCallback(() => {
    if (tokenBufferRef.current) {
      const batch = tokenBufferRef.current
      tokenBufferRef.current = ""
      appendStreamBatch(batch)
    }
    rafIdRef.current = null
  }, [appendStreamBatch])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const sendMessage = useCallback(
    async (questionText: string) => {
      const trimmed = questionText.trim()
      if (!trimmed || isGenerating) return

      let targetConvId = activeConversationId

      // 1. If in "New Chat" state without an active ID, create one
      if (!targetConvId) {
        try {
          const created = await ConversationService.createConversation({ title: "New Chat" })
          targetConvId = created.id
          setActiveConversationId(created.id)
          queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
        } catch (err) {
          console.error("Failed to create conversation:", err)
          return
        }
      }

      const convId = targetConvId

      // 2. Add optimistic user message to local cache
      const tempUserMsg: ChatMessage = {
        id: crypto.randomUUID(),
        conversation_id: convId,
        role: "user",
        content: trimmed,
        message_status: "COMPLETED",
        created_at: new Date().toISOString(),
      }

      queryClient.setQueryData<ChatMessage[]>(messagesQueryKey(convId), (prev = []) => [
        ...prev,
        tempUserMsg,
      ])

      // Optimistically update conversation's updated_at and bubble to top of sidebar
      queryClient.setQueryData<ConversationRead[]>(CONVERSATIONS_QUERY_KEY, (prev = []) => {
        const now = new Date().toISOString()
        const existing = prev.find((c) => c.id === convId)
        if (existing) {
          const updated = {
            ...existing,
            updated_at: now,
            message_count: (existing.message_count || 0) + 1,
          }
          return [updated, ...prev.filter((c) => c.id !== convId)]
        }
        return prev
      })

      // 3. Initialize streaming state
      const reqId = crypto.randomUUID()
      startGeneration(reqId, convId)

      const abortCtrl = new AbortController()
      abortControllerRef.current = abortCtrl

      let accumulatedAnswer = ""
      tokenBufferRef.current = ""

      // Schedule batch flush using requestAnimationFrame
      const scheduleTokenFlush = () => {
        if (rafIdRef.current === null) {
          rafIdRef.current = requestAnimationFrame(() => {
            flushTokenBuffer()
          })
        }
      }

      // 4. Execute streaming request
      await streamAnswer(
        buildAnswerPayload({
          question: trimmed,
          conversation_id: convId,
        }),
        {
          onStart: () => {
            setStreamStage("Searching knowledge base...")
          },
          onProgress: (prog) => {
            setStreamStage(prog.message)
          },
          onToken: (tok) => {
            accumulatedAnswer += tok
            tokenBufferRef.current += tok
            scheduleTokenFlush()
          },
          onComplete: (complete) => {
            // Cancel any pending RAF and flush remaining tokens immediately
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            tokenBufferRef.current = ""

            const finalAnswerText = complete.answer_text || accumulatedAnswer
            setStreamingAnswer(finalAnswerText)

            const assistantMsg: ChatMessage = {
              id: complete.message_id || crypto.randomUUID(),
              conversation_id: convId,
              role: "assistant",
              content: finalAnswerText,
              message_status: "COMPLETED",
              source_references: complete.source_references,
              diagnostics: complete.diagnostics,
              created_at: new Date().toISOString(),
            }

            queryClient.setQueryData<ChatMessage[]>(messagesQueryKey(convId), (prev = []) => [
              ...prev,
              assistantMsg,
            ])

            queryClient.invalidateQueries({ queryKey: messagesQueryKey(convId) })
            queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
            stopGeneration()
          },
          onTitle: () => {
            // Refresh conversation titles in the sidebar
            queryClient.invalidateQueries({ queryKey: CONVERSATIONS_QUERY_KEY })
          },
          onCancelled: () => {
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            tokenBufferRef.current = ""

            if (accumulatedAnswer) {
              const partialMsg: ChatMessage = {
                id: crypto.randomUUID(),
                conversation_id: convId,
                role: "assistant",
                content: accumulatedAnswer + " *(Generation stopped)*",
                message_status: "CANCELLED",
                created_at: new Date().toISOString(),
              }
              queryClient.setQueryData<ChatMessage[]>(messagesQueryKey(convId), (prev = []) => [
                ...prev,
                partialMsg,
              ])
            }
            queryClient.invalidateQueries({ queryKey: messagesQueryKey(convId) })
            stopGeneration()
          },
          onError: (err) => {
            if (rafIdRef.current !== null) {
              cancelAnimationFrame(rafIdRef.current)
              rafIdRef.current = null
            }
            tokenBufferRef.current = ""

            console.error("Streaming error:", err)
            const errorMsg: ChatMessage = {
              id: crypto.randomUUID(),
              conversation_id: convId,
              role: "assistant",
              content:
                accumulatedAnswer ||
                "An error occurred while generating the answer. Please verify local Ollama is running and try again.",
              message_status: "FAILED",
              created_at: new Date().toISOString(),
            }
            queryClient.setQueryData<ChatMessage[]>(messagesQueryKey(convId), (prev = []) => [
              ...prev,
              errorMsg,
            ])
            stopGeneration()
          },
        },
        abortCtrl.signal
      )
    },
    [
      activeConversationId,
      isGenerating,
      queryClient,
      setActiveConversationId,
      startGeneration,
      flushTokenBuffer,
      setStreamingAnswer,
      setStreamStage,
      stopGeneration,
    ]
  )

  const stopCurrentGeneration = useCallback(async () => {
    if (activeRequestId) {
      try {
        await AnswerService.cancelGeneration(activeRequestId)
      } catch (err) {
        console.warn("Cancellation API error:", err)
      }
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current)
      rafIdRef.current = null
    }
    tokenBufferRef.current = ""
    stopGeneration()
  }, [activeRequestId, stopGeneration])

  return {
    sendMessage,
    stopGeneration: stopCurrentGeneration,
    isGenerating,
    streamingAnswer,
    streamingStage,
    activeStreamingConversationId,
  }
}
