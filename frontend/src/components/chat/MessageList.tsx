import React, { useEffect, useRef, useState, useCallback } from "react"
import { MessageBubble } from "./MessageBubble"
import { StreamingIndicator } from "./StreamingIndicator"
import { useChatStore } from "../../stores/chat.store"
import type { ChatMessage } from "../../types/message"

interface MessageListProps {
  messages: ChatMessage[]
  onRetryLast?: () => void
}

export const MessageList: React.FC<MessageListProps> = ({ messages, onRetryLast }) => {
  const {
    isGenerating,
    activeStreamingConversationId,
    activeConversationId,
    streamingAnswer,
    streamingStage,
  } = useChatStore()

  const containerRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [userScrolledUp, setUserScrolledUp] = useState(false)

  const isCurrentConversationStreaming =
    isGenerating &&
    activeStreamingConversationId !== null &&
    activeStreamingConversationId === activeConversationId

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight)
    // If user is more than 120px from bottom, consider them reading history
    setUserScrolledUp(distanceFromBottom > 120)
  }, [])

  useEffect(() => {
    if (!userScrolledUp && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, streamingAnswer, isCurrentConversationStreaming, userScrolledUp])

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 overflow-y-auto px-2 sm:px-4 py-4 divide-y divide-border/20"
    >
      <div className="max-w-4xl mx-auto w-full flex flex-col">
        {messages.map((msg, idx) => (
          <MessageBubble
            key={msg.id || idx}
            message={msg}
            onRetry={idx === messages.length - 1 ? onRetryLast : undefined}
          />
        ))}

        {/* Real-time Streaming message */}
        {isCurrentConversationStreaming && (
          <div className="pt-2">
            {streamingAnswer ? (
              <MessageBubble
                message={{
                  id: "streaming-active",
                  conversation_id: activeConversationId || "",
                  role: "assistant",
                  content: streamingAnswer,
                  message_status: "IN_PROGRESS",
                  created_at: new Date().toISOString(),
                }}
                isStreaming
              />
            ) : (
              <div className="px-6 py-4">
                <StreamingIndicator stage={streamingStage} />
              </div>
            )}
          </div>
        )}

        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  )
}

