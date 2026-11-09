import React, { memo } from "react"
import { Sparkles, User, AlertCircle } from "lucide-react"
import { MarkdownRenderer } from "../messages/MarkdownRenderer"
import { MessageActions } from "../messages/MessageActions"
import { ErrorBoundary } from "../ui/ErrorBoundary"
import type { ChatMessage } from "../../types/message"

interface MessageBubbleProps {
  message: ChatMessage
  isStreaming?: boolean
  onRetry?: () => void
}

const MessageBubbleComponent: React.FC<MessageBubbleProps> = ({
  message,
  isStreaming = false,
  onRetry,
}) => {
  const isUser = message.role === "user"
  const isFailed = message.message_status === "FAILED"

  return (
    <div
      className={`group flex w-full py-4 px-4 sm:px-6 transition-colors ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`flex gap-3 sm:gap-4 max-w-3xl w-full ${
          isUser ? "flex-row-reverse" : "flex-row"
        }`}
      >
        {/* Avatar */}
        <div
          className={`flex items-center justify-center w-8 h-8 rounded-xl flex-shrink-0 mt-0.5 shadow-xs ${
            isUser
              ? "bg-muted text-muted-foreground"
              : isFailed
              ? "bg-red-500/10 text-red-400"
              : "bg-gradient-to-tr from-indigo-500 to-purple-600 text-white"
          }`}
        >
          {isUser ? (
            <User className="w-4 h-4" />
          ) : isFailed ? (
            <AlertCircle className="w-4 h-4" />
          ) : (
            <Sparkles className="w-4 h-4" />
          )}
        </div>

        {/* Message Body */}
        <div className={`flex flex-col flex-1 min-w-0 ${isUser ? "items-end" : "items-start"}`}>
          {/* Sender label */}
          <span className="text-[11px] font-semibold text-muted-foreground mb-1 select-none">
            {isUser ? "You" : "Nexia"}
          </span>

          {/* Content */}
          <div
            className={`w-full ${
              isUser
                ? "bg-muted/70 text-foreground px-4 py-2.5 rounded-2xl max-w-xl text-[15px] leading-relaxed"
                : "text-foreground"
            }`}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap break-words">{message.content}</p>
            ) : (
              <div>
                <MarkdownRenderer content={message.content} />
                {isStreaming && <span className="cursor-pulse" />}
              </div>
            )}
          </div>

          {/* Message Actions */}
          {!isStreaming && (
            <MessageActions
              content={message.content}
              role={message.role}
              onRetry={onRetry}
              showRetry={!isUser}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export const MessageBubble = memo(
  (props: MessageBubbleProps) => (
    <ErrorBoundary fallbackTitle="Failed to render message">
      <MessageBubbleComponent {...props} />
    </ErrorBoundary>
  ),
  (prev, next) => {
    return (
      prev.message.id === next.message.id &&
      prev.message.content === next.message.content &&
      prev.message.message_status === next.message.message_status &&
      prev.isStreaming === next.isStreaming &&
      prev.onRetry === next.onRetry
    )
  }
)

MessageBubble.displayName = "MessageBubble"
