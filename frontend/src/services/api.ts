const BASE_URL = import.meta.env.VITE_API_URL || ""

export class ApiError extends Error {
  status: number
  data?: any

  constructor(status: number, message: string, data?: any) {
    super(message)
    this.name = "ApiError"
    this.status = status
    this.data = data
  }
}

export async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`
  const headers = new Headers(options.headers || {})

  if (!headers.has("Content-Type") && options.body && typeof options.body === "string") {
    headers.set("Content-Type", "application/json")
  }

  const response = await fetch(url, {
    ...options,
    headers,
  })

  if (response.status === 204) {
    return null as unknown as T
  }

  let data: any
  const contentType = response.headers.get("content-type")
  if (contentType && contentType.includes("application/json")) {
    try {
      data = await response.json()
    } catch {
      data = null
    }
  } else {
    data = await response.text()
  }

  if (!response.ok) {
    const errorMsg = data?.detail || data?.message || `HTTP ${response.status}: ${response.statusText}`
    throw new ApiError(response.status, errorMsg, data)
  }

  return data as T
}

