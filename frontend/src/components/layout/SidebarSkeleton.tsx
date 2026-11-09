import React from "react"
import { Skeleton } from "../ui/Skeleton"

export const SidebarSkeleton: React.FC = () => {
  return (
    <div className="flex flex-col gap-2 p-2 w-full animate-in fade-in duration-200">
      <div className="px-3 py-1.5">
        <Skeleton className="h-3 w-16 rounded" />
      </div>
      {[92, 75, 85, 60, 80].map((widthPct, idx) => (
        <div
          key={idx}
          className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-sidebar-hover/20"
        >
          <Skeleton className="w-4 h-4 rounded-md flex-shrink-0" />
          <Skeleton className="h-4 rounded-md flex-1" style={{ maxWidth: `${widthPct}%` }} />
        </div>
      ))}
    </div>
  )
}

