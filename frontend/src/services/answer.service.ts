import { request } from "./api"
import type { AnswerRequest, AnswerResponse } from "../types/api"

const API_V1 = "/api/v1"

export const AnswerService = {
  async generateAnswer(req: AnswerRequest): Promise<AnswerResponse> {
    return request<AnswerResponse>(`${API_V1}/retrieval/answer`, {
      method: "POST",
      body: JSON.stringify(req),
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

