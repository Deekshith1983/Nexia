import React from "react"
import { Sparkles } from "lucide-react"

interface StreamingIndicatorProps {
  stage?: string | null
}

export const StreamingIndicator: React.FC<StreamingIndicatorProps> = ({ stage }) => {
  return (
    <div className="flex items-center gap-3 py-2 text-sm text-muted-foreground animate-in fade-in duration-200">
      <div className="relative flex items-center justify-center w-7 h-7 rounded-xl bg-indigo-500/10 text-indigo-400">
        <Sparkles className="w-4 h-4 animate-spin text-indigo-400" style={{ animationDuration: "3s" }} />
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-muted-foreground">
          {stage || "Thinking..."}
        </span>
        <span className="flex gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/80 animate-bounce" style={{ animationDelay: "0ms" }} />
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/80 animate-bounce" style={{ animationDelay: "150ms" }} />
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400/80 animate-bounce" style={{ animationDelay: "300ms" }} />
        </span>
      </div>
    </div>
  )
}

