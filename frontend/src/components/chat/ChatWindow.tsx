import React, { useState, useCallback } from "react"
import { useChatStore } from "../../stores/chat.store"
import { useConversationMessages } from "../../hooks/useConversationMessages"
import { useChat } from "../../hooks/useChat"
import { WelcomeScreen } from "./WelcomeScreen"
import { MessageList } from "./MessageList"
import { MessageComposer } from "./MessageComposer"
import { MessageSkeleton } from "./MessageSkeleton"
import { ErrorBoundary } from "../ui/ErrorBoundary"

export const ChatWindow: React.FC = () => {
  const activeConversationId = useChatStore((s) => s.activeConversationId)
  const { messages, isLoading } = useConversationMessages(activeConversationId)
  const { sendMessage, stopGeneration, isGenerating } = useChat()

  const [composerPrompt, setComposerPrompt] = useState("")

  const handleSelectPrompt = useCallback(
    (prompt: string) => {
      sendMessage(prompt)
    },
    [sendMessage]
  )

  const handleRetryLast = useCallback(() => {
    const userMessages = messages.filter((m) => m.role === "user")
    if (userMessages.length > 0) {
      const lastUserMsg = userMessages[userMessages.length - 1]
      sendMessage(lastUserMsg.content)
    }
  }, [messages, sendMessage])

  const showWelcome = !activeConversationId && messages.length === 0

  return (
    <ErrorBoundary fallbackTitle="Chat interface error" fallbackMessage="An error occurred in the conversation view.">
      <div className="flex flex-col flex-1 h-full min-w-0 overflow-hidden relative">
        {isLoading ? (
          <MessageSkeleton />
        ) : showWelcome ? (
          <div className="flex flex-col flex-1 overflow-y-auto">
            <WelcomeScreen onSelectPrompt={handleSelectPrompt} />
          </div>
        ) : (
          <MessageList messages={messages} onRetryLast={handleRetryLast} />
        )}

        {/* Floating Bottom Composer */}
        <MessageComposer
          onSend={(msg) => {
            setComposerPrompt("")
            sendMessage(msg)
          }}
          onStop={stopGeneration}
          isGenerating={isGenerating}
          initialPrompt={composerPrompt}
        />
      </div>
    </ErrorBoundary>
  )
}
