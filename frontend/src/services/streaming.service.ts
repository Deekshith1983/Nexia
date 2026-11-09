import { buildAnswerPayload } from "../config/retrieval.config"
import type {
  AnswerRequest,
  SSECancelledEvent,
  SSECitationEvent,
  SSECompleteEvent,
  SSEErrorEvent,
  SSEProgressEvent,
  SSEStartEvent,
  SSETitleEvent,
} from "../types/api"

export interface StreamCallbacks {
  onStart?: (event: SSEStartEvent) => void
  onProgress?: (event: SSEProgressEvent) => void
  onToken?: (token: string) => void
  onCitation?: (event: SSECitationEvent) => void
  onComplete?: (event: SSECompleteEvent) => void
  onTitle?: (event: SSETitleEvent) => void
  onError?: (error: Error | SSEErrorEvent) => void
  onCancelled?: (event: SSECancelledEvent) => void
}

const API_V1 = "/api/v1"

export async function streamAnswer(
  request: AnswerRequest,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const url = `${API_V1}/retrieval/answer/stream`
  const payload = buildAnswerPayload(request)

  let response: Response
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    })
  } catch (err: any) {
    if (signal?.aborted) {
      callbacks.onCancelled?.({ request_id: "", reason: "Client aborted" })
      return
    }
    callbacks.onError?.(err)
    return
  }

  if (!response.ok) {
    const errorText = await response.text()
    callbacks.onError?.(new Error(`Server error (${response.status}): ${errorText || response.statusText}`))
    return
  }

  if (!response.body) {
    callbacks.onError?.(new Error("Response body is not readable"))
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process complete SSE messages separated by double newlines
      const events = buffer.split(/\r?\n\r?\n/)
      // Keep the last partial event fragment in the buffer
      buffer = events.pop() || ""

      for (const eventBlock of events) {
        if (!eventBlock.trim()) continue

        let eventType = "message"
        let dataStr = ""

        const lines = eventBlock.split(/\r?\n/)
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith("data:")) {
            dataStr = line.slice(5).trim()
          }
        }

        if (!dataStr) continue

        try {
          const parsed = JSON.parse(dataStr)

          switch (eventType) {
            case "start":
              callbacks.onStart?.(parsed as SSEStartEvent)
              break
            case "progress":
              callbacks.onProgress?.(parsed as SSEProgressEvent)
              break
            case "token":
              callbacks.onToken?.(parsed.token || "")
              break
            case "citation":
              callbacks.onCitation?.(parsed as SSECitationEvent)
              break
            case "complete":
              callbacks.onComplete?.(parsed as SSECompleteEvent)
              break
            case "title":
              callbacks.onTitle?.(parsed as SSETitleEvent)
              break
            case "cancelled":
              callbacks.onCancelled?.(parsed as SSECancelledEvent)
              break
            case "error":
              callbacks.onError?.(parsed as SSEErrorEvent)
              break
            default:
              break
          }
        } catch {
          // Ignore JSON parse errors for non-data lines
        }
      }
    }
  } catch (err: any) {
    if (signal?.aborted) {
      callbacks.onCancelled?.({ request_id: "", reason: "Client aborted" })
      return
    }
    callbacks.onError?.(err)
  } finally {
    reader.releaseLock()
  }
}

