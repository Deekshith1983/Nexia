import React from "react"
import { Skeleton } from "../ui/Skeleton"

export const MessageSkeleton: React.FC = () => {
  return (
    <div className="flex-1 overflow-hidden px-4 sm:px-6 py-6 divide-y divide-border/10 max-w-4xl mx-auto w-full">
      {/* 1. User Message Skeleton */}
      <div className="flex justify-end py-4">
        <div className="flex flex-row-reverse gap-3.5 max-w-xl w-full">
          <Skeleton className="w-8 h-8 rounded-xl flex-shrink-0" />
          <div className="flex flex-col items-end gap-2 flex-1">
            <Skeleton className="h-3 w-10 rounded" />
            <Skeleton className="h-10 w-3/4 rounded-2xl" />
          </div>
        </div>
      </div>

      {/* 2. Assistant Message Skeleton */}
      <div className="flex justify-start py-4">
        <div className="flex flex-row gap-3.5 max-w-2xl w-full">
          <Skeleton className="w-8 h-8 rounded-xl flex-shrink-0" />
          <div className="flex flex-col items-start gap-2.5 flex-1">
            <Skeleton className="h-3 w-14 rounded" />
            <Skeleton className="h-4 w-full rounded" />
            <Skeleton className="h-4 w-5/6 rounded" />
            <Skeleton className="h-4 w-2/3 rounded" />
            <div className="mt-2 flex gap-2">
              <Skeleton className="h-6 w-16 rounded-md" />
              <Skeleton className="h-6 w-16 rounded-md" />
            </div>
          </div>
        </div>
      </div>

      {/* 3. Second User Message Skeleton */}
      <div className="flex justify-end py-4">
        <div className="flex flex-row-reverse gap-3.5 max-w-xl w-full">
          <Skeleton className="w-8 h-8 rounded-xl flex-shrink-0" />
          <div className="flex flex-col items-end gap-2 flex-1">
            <Skeleton className="h-3 w-10 rounded" />
            <Skeleton className="h-12 w-4/5 rounded-2xl" />
          </div>
        </div>
      </div>
    </div>
  )
}

