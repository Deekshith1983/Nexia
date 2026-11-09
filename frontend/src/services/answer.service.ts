import { request } from "./api"
import { buildAnswerPayload } from "../config/retrieval.config"
import type { AnswerRequest, AnswerResponse } from "../types/api"

const API_V1 = "/api/v1"

export const AnswerService = {
  async generateAnswer(req: AnswerRequest): Promise<AnswerResponse> {
    const payload = buildAnswerPayload(req)
    return request<AnswerResponse>(`${API_V1}/retrieval/answer`, {
      method: "POST",
      body: JSON.stringify(payload),
    })
  },

  async cancelGeneration(requestId: string): Promise<{ request_id: string; cancelled: boolean }> {
    return request<{ request_id: string; cancelled: boolean }>(
      `${API_V1}/retrieval/cancel/${requestId}`,
      {
        method: "POST",
      }
    )
  },
}
