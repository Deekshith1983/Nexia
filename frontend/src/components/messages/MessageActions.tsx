import React, { useState } from "react"
import { Check, Copy, RefreshCw } from "lucide-react"

interface MessageActionsProps {
  content: string
  role: "user" | "assistant" | "system"
  onRetry?: () => void
  showRetry?: boolean
}

export const MessageActions: React.FC<MessageActionsProps> = ({
  content,
  role,
  onRetry,
  showRetry = false,
}) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy:", err)
    }
  }

  return (
    <div className="flex items-center gap-1 mt-2 text-muted-foreground opacity-80 hover:opacity-100 transition-opacity">
      <button
        onClick={handleCopy}
        type="button"
        title="Copy text"
        aria-label="Copy message text"
        className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
      >
        {copied ? (
          <Check className="w-3.5 h-3.5 text-emerald-400" />
        ) : (
          <Copy className="w-3.5 h-3.5" />
        )}
      </button>

      {role === "assistant" && showRetry && onRetry && (
        <button
          onClick={onRetry}
          type="button"
          title="Retry response"
          aria-label="Retry response generation"
          className="p-1.5 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  )
}

