export const RETRIEVAL_CONFIG = {
  top_k_per_document: 3,
  min_confidence: 0.4,
  max_evidence_passages: 9,
} as const;

/**
 * Builds a standardized answer request payload enforcing Nexia's canonical retrieval configuration.
 * Caller-provided values for retrieval configuration are strictly overridden by RETRIEVAL_CONFIG.
 */
export function buildAnswerPayload<T extends { question: string }>(
  requestData: T
): T & typeof RETRIEVAL_CONFIG {
  return {
    ...requestData,
    top_k_per_document: RETRIEVAL_CONFIG.top_k_per_document,
    min_confidence: RETRIEVAL_CONFIG.min_confidence,
    max_evidence_passages: RETRIEVAL_CONFIG.max_evidence_passages,
  };
}

