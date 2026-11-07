import React, { useState } from "react"
import { useChatStore } from "../../stores/chat.store"
import { useConversationMessages } from "../../hooks/useConversationMessages"
import { useChat } from "../../hooks/useChat"
import { WelcomeScreen } from "./WelcomeScreen"
import { MessageList } from "./MessageList"
import { MessageComposer } from "./MessageComposer"

export const ChatWindow: React.FC = () => {
  const { activeConversationId } = useChatStore()
  const { messages, isLoading } = useConversationMessages(activeConversationId)
  const { sendMessage, stopGeneration, isGenerating } = useChat()

  const [composerPrompt, setComposerPrompt] = useState("")

  const handleSelectPrompt = (prompt: string) => {
    sendMessage(prompt)
  }

  const handleRetryLast = () => {
    const userMessages = messages.filter((m) => m.role === "user")
    if (userMessages.length > 0) {
      const lastUserMsg = userMessages[userMessages.length - 1]
      sendMessage(lastUserMsg.content)
    }
  }

  const showWelcome = !activeConversationId && messages.length === 0

  return (
    <div className="flex flex-col flex-1 h-full min-w-0 overflow-hidden relative">
      {isLoading ? (
        <div className="flex flex-col items-center justify-center flex-1 gap-3 text-muted-foreground">
          <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-xs">Loading conversation history...</p>
        </div>
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
  )
}

