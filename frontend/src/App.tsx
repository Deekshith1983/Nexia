import { useEffect } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ChatPage } from "./pages/ChatPage"
import { useChatStore } from "./stores/chat.store"

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

export function App() {
  const { theme } = useChatStore()

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
  }, [theme])

  return (
    <QueryClientProvider client={queryClient}>
      <ChatPage />
    </QueryClientProvider>
  )
}

export default App
