import React from "react"
import { describe, it, expect, vi } from "vitest"
import "@testing-library/jest-dom/vitest"
import { render, screen, fireEvent } from "@testing-library/react"
import { WelcomeScreen } from "../components/chat/WelcomeScreen"
import { MessageBubble } from "../components/chat/MessageBubble"
import { MessageComposer } from "../components/chat/MessageComposer"
import { ConfirmModal } from "../components/ui/ConfirmModal"

describe("Chat Components", () => {
  it("renders WelcomeScreen with heading and suggested prompts", () => {
    const onSelectPrompt = vi.fn()
    render(<WelcomeScreen onSelectPrompt={onSelectPrompt} />)

    expect(screen.getByText("How can I help you today?")).toBeInTheDocument()
    expect(
      screen.getByText(/Ask anything. Explore ideas, understand complex topics/i)
    ).toBeInTheDocument()

    const promptBtn = screen.getByText(/Who was Shivaji Maharaj/i)
    expect(promptBtn).toBeInTheDocument()

    fireEvent.click(promptBtn)
    expect(onSelectPrompt).toHaveBeenCalledWith(
      expect.stringContaining("Shivaji Maharaj")
    )
  })

  it("renders MessageBubble for user messages", () => {
    render(
      <MessageBubble
        message={{
          id: "msg-1",
          conversation_id: "conv-1",
          role: "user",
          content: "What is quantum mechanics?",
          message_status: "COMPLETED",
          created_at: new Date().toISOString(),
        }}
      />
    )

    expect(screen.getByText("You")).toBeInTheDocument()
    expect(screen.getByText("What is quantum mechanics?")).toBeInTheDocument()
  })

  it("renders MessageBubble for assistant answers with markdown & Wikipedia link", () => {
    render(
      <MessageBubble
        message={{
          id: "msg-2",
          conversation_id: "conv-1",
          role: "assistant",
          content:
            "Samudragupta was a Gupta ruler. [Allahabad Pillar Inscription](https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription)",
          message_status: "COMPLETED",
          created_at: new Date().toISOString(),
        }}
      />
    )

    expect(screen.getByText("Nexia")).toBeInTheDocument()
    expect(screen.getByText(/Samudragupta was a Gupta ruler/i)).toBeInTheDocument()
    const link = screen.getByRole("link", { name: /Allahabad Pillar Inscription/i })
    expect(link).toHaveAttribute("href", "https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription")
  })

  it("renders MessageComposer and handles submission and stop generation", () => {
    const onSend = vi.fn()
    const onStop = vi.fn()

    const { rerender } = render(
      <MessageComposer onSend={onSend} onStop={onStop} isGenerating={false} />
    )

    const textarea = screen.getByPlaceholderText(/Ask Nexia anything/i)
    fireEvent.change(textarea, { target: { value: "Tell me about Ashoka" } })

    const sendBtn = screen.getByLabelText("Send question")
    fireEvent.click(sendBtn)
    expect(onSend).toHaveBeenCalledWith("Tell me about Ashoka")

    // When generating:
    rerender(<MessageComposer onSend={onSend} onStop={onStop} isGenerating={true} />)
    const stopBtn = screen.getByLabelText("Stop generation")
    expect(stopBtn).toBeInTheDocument()
    fireEvent.click(stopBtn)
    expect(onStop).toHaveBeenCalled()
  })

  it("renders ConfirmModal and calls callbacks", () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()

    render(
      <ConfirmModal
        isOpen={true}
        title="Delete Item?"
        message="Are you sure?"
        onConfirm={onConfirm}
        onCancel={onCancel}
        isDestructive
      />
    )

    expect(screen.getByText("Delete Item?")).toBeInTheDocument()
    expect(screen.getByText("Are you sure?")).toBeInTheDocument()

    fireEvent.click(screen.getByText("Cancel"))
    expect(onCancel).toHaveBeenCalled()

    fireEvent.click(screen.getByText("Confirm"))
    expect(onConfirm).toHaveBeenCalled()
  })
})

