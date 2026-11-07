import React, { useState } from "react"
import { Check, Copy } from "lucide-react"

interface CodeBlockProps {
  language?: string
  value: string
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ language, value }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy code:", err)
    }
  }

  return (
    <div className="relative my-4 overflow-hidden rounded-xl bg-neutral-900 border border-neutral-800 text-neutral-100 font-mono text-sm shadow-md">
      <div className="flex items-center justify-between px-4 py-2 bg-neutral-950/60 border-b border-neutral-800/80 text-xs text-neutral-400">
        <span className="font-medium tracking-wide uppercase">{language || "code"}</span>
        <button
          onClick={handleCopy}
          type="button"
          aria-label="Copy code"
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium hover:bg-neutral-800 hover:text-white transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="p-4 overflow-x-auto leading-relaxed">
        <code>{value}</code>
      </div>
    </div>
  )
}

