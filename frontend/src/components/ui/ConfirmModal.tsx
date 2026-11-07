import React from "react"
import { AlertTriangle } from "lucide-react"

interface ConfirmModalProps {
  isOpen: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  isDestructive?: boolean
  onConfirm: () => void
  onCancel: () => void
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title,
  message,
  confirmText = "Confirm",
  cancelText = "Cancel",
  isDestructive = false,
  onConfirm,
  onCancel,
}) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-md p-6 bg-card text-card-foreground border border-border rounded-2xl shadow-2xl scale-100 transition-all">
        <div className="flex items-start gap-4">
          <div
            className={`p-3 rounded-full flex-shrink-0 ${
              isDestructive ? "bg-red-500/10 text-red-500" : "bg-neutral-800 text-neutral-200"
            }`}
          >
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed">{message}</p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 mt-6">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 text-sm font-medium rounded-xl hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          >
            {cancelText}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`px-4 py-2 text-sm font-medium rounded-xl transition-colors shadow-sm ${
              isDestructive
                ? "bg-red-600 hover:bg-red-700 text-white"
                : "bg-primary hover:bg-primary/90 text-primary-foreground"
            }`}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  )
}

