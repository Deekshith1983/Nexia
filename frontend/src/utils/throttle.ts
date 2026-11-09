/**
 * Throttles a function using requestAnimationFrame or a fixed interval.
 */
export function throttle<T extends (...args: any[]) => void>(
  func: T,
  limitMs: number
): (...args: Parameters<T>) => void {
  let inThrottle = false
  let lastArgs: Parameters<T> | null = null

  return function (this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
        if (lastArgs) {
          func.apply(this, lastArgs)
          lastArgs = null
        }
      }, limitMs)
    } else {
      lastArgs = args
    }
  }
}

/**
 * Throttles execution to browser animation frames (16.6ms at 60Hz).
 */
export function rafThrottle<T extends (...args: any[]) => void>(
  callback: T
): (...args: Parameters<T>) => void {
  let requestId: number | null = null

  return function (this: any, ...args: Parameters<T>) {
    if (requestId === null) {
      requestId = requestAnimationFrame(() => {
        requestId = null
        callback.apply(this, args)
      })
    }
  }
}

