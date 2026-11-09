import React, { memo, useMemo } from "react"
import ReactMarkdown from "react-markdown"
import type { Components } from "react-markdown"
import remarkGfm from "remark-gfm"
import { CodeBlock } from "./CodeBlock"
import { ExternalLink } from "lucide-react"
import { ErrorBoundary } from "../ui/ErrorBoundary"

interface MarkdownRendererProps {
  content: string
}

const REMARK_PLUGINS = [remarkGfm]

const InnerMarkdown: React.FC<MarkdownRendererProps> = memo(({ content }) => {
  const components: Components = useMemo(
    () => ({
      code({ className, children, ...props }) {
        const match = /language-(\w+)/.exec(className || "")
        const isInline = !match && !String(children).includes("\n")

        if (isInline) {
          return (
            <code
              className="px-1.5 py-0.5 rounded-md bg-muted text-foreground font-mono text-[13px] border border-border/50"
              {...props}
            >
              {children}
            </code>
          )
        }

        return (
          <CodeBlock
            language={match ? match[1] : undefined}
            value={String(children).replace(/\n$/, "")}
          />
        )
      },
      a({ href, children, ...props }) {
        const isWikipedia = href?.includes("wikipedia.org")
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className={`inline-flex items-center gap-1 font-medium transition-colors ${
              isWikipedia
                ? "text-sky-400 hover:text-sky-300 underline decoration-sky-400/40 hover:decoration-sky-300"
                : "text-primary underline hover:text-primary/80"
            }`}
            {...props}
          >
            <span>{children}</span>
            {isWikipedia && <ExternalLink className="w-3.5 h-3.5 opacity-75 inline" />}
          </a>
        )
      },
    }),
    []
  )

  return (
    <div className="markdown-content text-[15px] text-foreground/90 leading-relaxed break-words">
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  )
})

InnerMarkdown.displayName = "InnerMarkdown"

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = memo(({ content }) => {
  return (
    <ErrorBoundary fallbackMessage="Unable to format this markdown content. Displaying plain text:">
      <InnerMarkdown content={content} />
    </ErrorBoundary>
  )
})

MarkdownRenderer.displayName = "MarkdownRenderer"
