import { describe, it, expect, vi, beforeEach } from "vitest"
import { RETRIEVAL_CONFIG, buildAnswerPayload } from "../config/retrieval.config"
import { AnswerService } from "../services/answer.service"
import { streamAnswer } from "../services/streaming.service"
import * as api from "../services/api"

describe("Frontend Retrieval Configuration Enforcement", () => {
  it("defines exact canonical retrieval configuration constants", () => {
    expect(RETRIEVAL_CONFIG).toEqual({
      top_k_per_document: 3,
      min_confidence: 0.4,
      max_evidence_passages: 9,
    })
  })

  it("1. Frontend request with no retrieval parameters: all three values are included", () => {
    const payload = buildAnswerPayload({ question: "What is quantum computing?" })
    expect(payload).toEqual({
      question: "What is quantum computing?",
      top_k_per_document: 3,
      min_confidence: 0.4,
      max_evidence_passages: 9,
    })
  })

  it("2. Frontend request with incorrect retrieval parameters: required values override them", () => {
    const payload = buildAnswerPayload({
      question: "Explain relativity",
      top_k_per_document: 10 as any,
      min_confidence: 0.99 as any,
      max_evidence_passages: 1 as any,
    })

    expect(payload.top_k_per_document).toBe(3)
    expect(payload.min_confidence).toBe(0.4)
    expect(payload.max_evidence_passages).toBe(9)
  })

  it("3. Normal answer endpoint: receives and sends the required values", async () => {
    const requestSpy = vi.spyOn(api, "request").mockResolvedValueOnce({
      question: "What is entropy?",
      answer_text: "Entropy is a measure of disorder.",
      answer_status: "ANSWERED",
      source_references: [],
      conflicts: [],
      sufficiency_rating: "SUFFICIENT",
      retrieval_diagnostics: {} as any,
    })

    await AnswerService.generateAnswer({
      question: "What is entropy?",
      top_k_per_document: 100 as any, // Should be overridden
    })

    expect(requestSpy).toHaveBeenCalledTimes(1)
    const callArgs = requestSpy.mock.calls[0]
    expect(callArgs[0]).toBe("/api/v1/retrieval/answer")
    const body = JSON.parse(callArgs[1]?.body as string)
    expect(body.top_k_per_document).toBe(3)
    expect(body.min_confidence).toBe(0.4)
    expect(body.max_evidence_passages).toBe(9)

    requestSpy.mockRestore()
  })

  it("4. Streaming answer endpoint: receives and sends the required values", async () => {
    const fetchSpy = vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      body: {
        getReader: () => ({
          read: vi.fn().mockResolvedValueOnce({ done: true, value: undefined }),
          releaseLock: vi.fn(),
        }),
      },
    } as any)

    await streamAnswer(
      {
        question: "Explain photosynthesis",
        top_k_per_document: 99 as any, // Should be overridden
        min_confidence: 0.85 as any,   // Should be overridden
        max_evidence_passages: 2 as any, // Should be overridden
      },
      {}
    )

    expect(fetchSpy).toHaveBeenCalledTimes(1)
    const fetchCall = fetchSpy.mock.calls[0]
    expect(fetchCall[0]).toBe("/api/v1/retrieval/answer/stream")
    const body = JSON.parse(fetchCall[1]?.body as string)
    expect(body.top_k_per_document).toBe(3)
    expect(body.min_confidence).toBe(0.4)
    expect(body.max_evidence_passages).toBe(9)

    fetchSpy.mockRestore()
  })

  it("5. Retry/regenerate: retains the required values through buildAnswerPayload", () => {
    const originalRequest = {
      question: "What happened at Waterloo?",
      conversation_id: "conv-123",
    }

    // First attempt
    const firstPayload = buildAnswerPayload(originalRequest)
    expect(firstPayload.top_k_per_document).toBe(3)
    expect(firstPayload.min_confidence).toBe(0.4)
    expect(firstPayload.max_evidence_passages).toBe(9)

    // Retry / regenerate attempt
    const retryPayload = buildAnswerPayload({
      ...originalRequest,
      // Even if retry payload attempts to pass rogue parameters
      top_k_per_document: 50 as any,
    })
    expect(retryPayload.top_k_per_document).toBe(3)
    expect(retryPayload.min_confidence).toBe(0.4)
    expect(retryPayload.max_evidence_passages).toBe(9)
  })
})

