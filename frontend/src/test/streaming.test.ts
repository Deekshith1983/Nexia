import { describe, it, expect, vi } from "vitest"
import { streamAnswer } from "../services/streaming.service"
import type { AnswerRequest } from "../types/api"

describe("StreamingService - SSE Parsing", () => {
  it("parses sequential SSE events correctly", async () => {
    const ssePayload = [
      'event: start\ndata: {"request_id":"req-123","model":"gemma3:4b","status":"STARTED"}\n\n',
      'event: progress\ndata: {"request_id":"req-123","stage":"retrieval","message":"Searching Wikipedia..."}\n\n',
      'event: token\ndata: {"request_id":"req-123","token":"Hello "}\n\n',
      'event: token\ndata: {"request_id":"req-123","token":"world!"}\n\n',
      'event: complete\ndata: {"request_id":"req-123","answer_text":"Hello world!","answer_status":"ANSWERED","source_references":[]}\n\n',
      'event: title\ndata: {"request_id":"req-123","title":"Hello World Greetings"}\n\n',
    ].join("")

    const mockResponse = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload))
          controller.close()
        },
      }),
      { status: 200, headers: { "Content-Type": "text/event-stream" } }
    )

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockResponse)

    const onStart = vi.fn()
    const onProgress = vi.fn()
    const onToken = vi.fn()
    const onComplete = vi.fn()
    const onTitle = vi.fn()

    const request: AnswerRequest = { question: "Hi" }

    await streamAnswer(request, {
      onStart,
      onProgress,
      onToken,
      onComplete,
      onTitle,
    })

    expect(onStart).toHaveBeenCalledWith(
      expect.objectContaining({ request_id: "req-123", model: "gemma3:4b" })
    )
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ stage: "retrieval", message: "Searching Wikipedia..." })
    )
    expect(onToken).toHaveBeenCalledTimes(2)
    expect(onToken).toHaveBeenNthCalledWith(1, "Hello ")
    expect(onToken).toHaveBeenNthCalledWith(2, "world!")
    expect(onComplete).toHaveBeenCalledWith(
      expect.objectContaining({ answer_text: "Hello world!", answer_status: "ANSWERED" })
    )
    expect(onTitle).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Hello World Greetings" })
    )
  })

  it("handles cancelled SSE events", async () => {
    const ssePayload = 'event: cancelled\ndata: {"request_id":"req-123","reason":"User cancelled"}\n\n'

    const mockResponse = new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(ssePayload))
          controller.close()
        },
      }),
      { status: 200 }
    )

    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockResponse)

    const onCancelled = vi.fn()
    await streamAnswer({ question: "Cancel test" }, { onCancelled })

    expect(onCancelled).toHaveBeenCalledWith(
      expect.objectContaining({ request_id: "req-123", reason: "User cancelled" })
    )
  })

  it("handles error SSE events and HTTP errors", async () => {
    const mockResponse = new Response("Model not available", { status: 503 })
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(mockResponse)

    const onError = vi.fn()
    await streamAnswer({ question: "Error test" }, { onError })

    expect(onError).toHaveBeenCalledWith(expect.any(Error))
  })
})

