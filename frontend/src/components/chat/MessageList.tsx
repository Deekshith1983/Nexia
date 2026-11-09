import React, { useEffect, useRef, useState, useCallback, useMemo } from "react"
import { ArrowDown } from "lucide-react"
import { MessageBubble } from "./MessageBubble"
import { StreamingIndicator } from "./StreamingIndicator"
import { useChatStore } from "../../stores/chat.store"
import { useVirtualizer } from "../../hooks/useVirtualizer"
import type { ChatMessage } from "../../types/message"

interface MessageListProps {
  messages: ChatMessage[]
  onRetryLast?: () => void
}

export const MessageList: React.FC<MessageListProps> = ({ messages, onRetryLast }) => {
  // Fine-grained Zustand selector subscriptions
  const isGenerating = useChatStore((s) => s.isGenerating)
  const activeStreamingConversationId = useChatStore((s) => s.activeStreamingConversationId)
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const streamingAnswer = useChatStore((s) => s.streamingAnswer)
  const streamingStage = useChatStore((s) => s.streamingStage)

  const containerRef = useRef<HTMLDivElement>(null)
  const [userScrolledUp, setUserScrolledUp] = useState(false)
  const isUserScrolledUpRef = useRef(false)

  const isCurrentConversationStreaming =
    isGenerating &&
    activeStreamingConversationId !== null &&
    activeStreamingConversationId === activeConversationId

  // Total count includes streaming item when active
  const totalCount = messages.length + (isCurrentConversationStreaming ? 1 : 0)

  // Virtualizer configuration
  const estimateSize = useCallback(
    (index: number) => {
      if (index === messages.length && isCurrentConversationStreaming) {
        return 120 // Estimated height for active streaming bubble
      }
      const msg = messages[index]
      if (!msg) return 80
      // User messages are typically shorter than assistant markdown
      if (msg.role === "user") return 65
      // Rough estimate based on content length
      const len = msg.content?.length || 0
      return Math.min(600, Math.max(90, Math.ceil(len / 3)))
    },
    [messages, isCurrentConversationStreaming]
  )

  const {
    getVirtualItems,
    getTotalSize,
    measureElement,
    scrollToBottom,
  } = useVirtualizer({
    count: totalCount,
    getScrollElement: useCallback(() => containerRef.current, []),
    estimateSize,
    overscan: 4,
  })

  // Detect user scroll position
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight)
    const isUp = distanceFromBottom > 120
    setUserScrolledUp(isUp)
    isUserScrolledUpRef.current = isUp
  }, [])

  // Auto-scroll to bottom only when user is NOT scrolled up
  useEffect(() => {
    if (!isUserScrolledUpRef.current && containerRef.current) {
      scrollToBottom(false)
    }
  }, [messages.length, streamingAnswer, isCurrentConversationStreaming, scrollToBottom])

  // Initial scroll to bottom on conversation change
  useEffect(() => {
    setUserScrolledUp(false)
    isUserScrolledUpRef.current = false
    const timer = setTimeout(() => {
      scrollToBottom(false)
    }, 50)
    return () => clearTimeout(timer)
  }, [activeConversationId, scrollToBottom])

  const virtualItems = getVirtualItems()
  const totalSize = getTotalSize()

  return (
    <div className="relative flex-1 min-h-0 overflow-hidden flex flex-col">
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-2 sm:px-4 py-4 divide-y divide-border/20 will-change-scroll"
      >
        <div className="max-w-4xl mx-auto w-full relative" style={{ height: `${totalSize}px` }}>
          {virtualItems.map((virtualItem) => {
            const isStreamingIndex = virtualItem.index === messages.length

            return (
              <div
                key={isStreamingIndex ? "streaming-bubble" : messages[virtualItem.index]?.id || virtualItem.index}
                data-index={virtualItem.index}
                ref={measureElement}
                className="absolute top-0 left-0 w-full"
                style={{
                  transform: `translateY(${virtualItem.start}px)`,
                }}
              >
                {isStreamingIndex ? (
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
                ) : (
                  <MessageBubble
                    message={messages[virtualItem.index]}
                    onRetry={
                      virtualItem.index === messages.length - 1 ? onRetryLast : undefined
                    }
                  />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Floating 'Scroll to bottom' button when user is scrolled up */}
      {userScrolledUp && (
        <button
          onClick={() => {
            setUserScrolledUp(false)
            isUserScrolledUpRef.current = false
            scrollToBottom(true)
          }}
          type="button"
          aria-label="Scroll to bottom"
          className="absolute bottom-4 right-6 z-20 flex items-center gap-1.5 px-3 py-2 rounded-full bg-card/90 backdrop-blur-md border border-border shadow-lg text-xs font-medium text-foreground hover:bg-muted transition-all duration-200 animate-in fade-in slide-in-from-bottom-2"
        >
          <ArrowDown className="w-3.5 h-3.5" />
          <span>Latest</span>
          {isCurrentConversationStreaming && (
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping ml-0.5" />
          )}
        </button>
      )}
    </div>
  )
}
