import React, { useState, useRef, useEffect } from "react"
import { MessageSquare, MoreHorizontal, Edit2, Trash2, Check, X } from "lucide-react"
import type { ConversationRead } from "../../types/conversation"

interface ConversationItemProps {
  conversation: ConversationRead
  isActive: boolean
  onSelect: () => void
  onRename: (newTitle: string) => Promise<void>
  onDelete: () => void
}

export const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  isActive,
  onSelect,
  onRename,
  onDelete,
}) => {
  const [isEditing, setIsEditing] = useState(false)
  const [editTitle, setEditTitle] = useState(conversation.title)
  const [menuOpen, setMenuOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setEditTitle(conversation.title)
  }, [conversation.title])

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    if (menuOpen) {
      document.addEventListener("mousedown", handleOutsideClick)
    }
    return () => document.removeEventListener("mousedown", handleOutsideClick)
  }, [menuOpen])

  const handleSaveRename = async () => {
    const trimmed = editTitle.trim()
    if (trimmed && trimmed !== conversation.title) {
      await onRename(trimmed)
    } else {
      setEditTitle(conversation.title)
    }
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSaveRename()
    } else if (e.key === "Escape") {
      setEditTitle(conversation.title)
      setIsEditing(false)
    }
  }

  return (
    <div
      className={`group relative flex items-center justify-between w-full px-3 py-2 rounded-xl text-sm transition-colors cursor-pointer select-none ${
        isActive
          ? "bg-sidebar-active text-foreground font-medium shadow-xs"
          : "text-sidebar-foreground hover:bg-sidebar-hover hover:text-foreground"
      }`}
      onClick={() => {
        if (!isEditing) onSelect()
      }}
    >
      <div className="flex items-center gap-2.5 min-w-0 flex-1">
        <MessageSquare className="w-4 h-4 flex-shrink-0 opacity-60 group-hover:opacity-100" />

        {isEditing ? (
          <div className="flex items-center gap-1.5 flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
            <input
              ref={inputRef}
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onKeyDown={handleKeyDown}
              className="w-full px-1.5 py-0.5 text-xs bg-background border border-border rounded-md text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            />
            <button
              onClick={handleSaveRename}
              className="p-1 hover:bg-neutral-800 rounded text-emerald-400"
              title="Save"
            >
              <Check className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => {
                setEditTitle(conversation.title)
                setIsEditing(false)
              }}
              className="p-1 hover:bg-neutral-800 rounded text-muted-foreground"
              title="Cancel"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <span className="truncate text-xs leading-normal">
            {conversation.title || "New Chat"}
          </span>
        )}
      </div>

      {!isEditing && (
        <div className="relative" ref={menuRef} onClick={(e) => e.stopPropagation()}>
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className={`p-1 rounded-md text-muted-foreground hover:text-foreground transition-opacity ${
              isActive || menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100"
            }`}
            title="Conversation options"
          >
            <MoreHorizontal className="w-3.5 h-3.5" />
          </button>

          {menuOpen && (
            <div className="absolute right-0 top-full mt-1 z-30 w-36 py-1 bg-card border border-border rounded-xl shadow-xl animate-in fade-in-50">
              <button
                onClick={() => {
                  setMenuOpen(false)
                  setIsEditing(true)
                }}
                className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-foreground hover:bg-muted transition-colors text-left"
              >
                <Edit2 className="w-3.5 h-3.5 text-muted-foreground" />
                <span>Rename</span>
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false)
                  onDelete()
                }}
                className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-red-500 hover:bg-red-500/10 transition-colors text-left"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Delete</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

