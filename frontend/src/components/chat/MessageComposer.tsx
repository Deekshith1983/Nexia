import React, { useState, useRef, useEffect, memo, useCallback } from "react"
import { ArrowUp, Square } from "lucide-react"

interface MessageComposerProps {
  onSend: (message: string) => void
  onStop: () => void
  isGenerating: boolean
  disabled?: boolean
  initialPrompt?: string
}

export const MessageComposer: React.FC<MessageComposerProps> = memo(
  ({ onSend, onStop, isGenerating, disabled = false, initialPrompt = "" }) => {
    const [text, setText] = useState(initialPrompt)
    const textareaRef = useRef<HTMLTextAreaElement>(null)
    const rafResizeRef = useRef<number | null>(null)

    useEffect(() => {
      if (initialPrompt) {
        setText(initialPrompt)
        if (textareaRef.current) {
          textareaRef.current.focus()
        }
      }
    }, [initialPrompt])

    // Auto-resize textarea height with requestAnimationFrame throttling
    const adjustTextareaHeight = useCallback(() => {
      if (rafResizeRef.current !== null) {
        cancelAnimationFrame(rafResizeRef.current)
      }
      rafResizeRef.current = requestAnimationFrame(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = "auto"
          textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
        }
        rafResizeRef.current = null
      })
    }, [])

    useEffect(() => {
      adjustTextareaHeight()
      return () => {
        if (rafResizeRef.current !== null) {
          cancelAnimationFrame(rafResizeRef.current)
        }
      }
    }, [text, adjustTextareaHeight])

    const handleSubmit = (e?: React.FormEvent) => {
      if (e) e.preventDefault()
      const trimmed = text.trim()
      if (!trimmed || isGenerating || disabled) return
      onSend(trimmed)
      setText("")
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto"
      }
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    }

    return (
      <div className="w-full max-w-4xl px-4 pb-4 sm:pb-6 mx-auto bg-gradient-to-t from-background via-background to-transparent pt-3">
        <form
          onSubmit={handleSubmit}
          className="relative flex flex-col w-full p-2 bg-card border border-border/80 focus-within:border-border rounded-2xl sm:rounded-3xl shadow-lg transition-all"
        >
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Nexia anything about history, science, or concepts..."
            rows={1}
            disabled={disabled}
            className="w-full px-3 py-2 text-[15px] bg-transparent text-foreground placeholder:text-muted-foreground/70 resize-none focus:outline-none min-h-[44px] max-h-[200px] leading-relaxed"
          />

          <div className="flex items-center justify-between px-2 pt-1 select-none">
            <div className="text-[11px] text-muted-foreground font-medium hidden sm:block">
              Shift + Return for new line
            </div>

            <div className="flex items-center gap-1 ml-auto">
              {isGenerating ? (
                <button
                  type="button"
                  onClick={onStop}
                  title="Stop generating"
                  aria-label="Stop generation"
                  className="flex items-center justify-center w-8 h-8 rounded-full bg-foreground text-background hover:opacity-90 transition-opacity shadow-sm"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!text.trim() || disabled}
                  title="Send question"
                  aria-label="Send question"
                  className={`flex items-center justify-center w-8 h-8 rounded-full transition-all ${
                    text.trim() && !disabled
                      ? "bg-foreground text-background hover:opacity-90 shadow-sm cursor-pointer"
                      : "bg-muted text-muted-foreground/50 cursor-not-allowed"
                  }`}
                >
                  <ArrowUp className="w-4 h-4 stroke-[2.5]" />
                </button>
              )}
            </div>
          </div>
        </form>

        <p className="mt-2 text-[11px] text-center text-muted-foreground/60 select-none">
          Nexia synthesizes knowledge grounded in Wikipedia. Verify important historical facts.
        </p>
      </div>
    )
  }
)

MessageComposer.displayName = "MessageComposer"
