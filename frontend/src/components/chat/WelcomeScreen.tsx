import React from "react"
import { Sparkles, Compass, BookOpen, Atom, Scroll } from "lucide-react"

interface WelcomeScreenProps {
  onSelectPrompt: (prompt: string) => void
}

const SUGGESTED_PROMPTS = [
  {
    icon: Scroll,
    title: "Historical Figures",
    prompt: "Who was Shivaji Maharaj and what was the significance of his coronation?",
  },
  {
    icon: BookOpen,
    title: "Ancient Inscriptions",
    prompt: "Describe the Allahabad Pillar inscription of Samudragupta and its historical context.",
  },
  {
    icon: Atom,
    title: "Quantum Physics",
    prompt: "What is quantum superposition and how do quantum computers use qubits?",
  },
  {
    icon: Compass,
    title: "Ancient Trade",
    prompt: "What were the primary trade goods exchanged at the ancient port of Muziris?",
  },
]

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSelectPrompt }) => {
  return (
    <div className="flex flex-col items-center justify-center flex-1 w-full max-w-3xl px-4 py-8 mx-auto my-auto text-center animate-in fade-in duration-300">
      {/* Brand Icon */}
      <div className="flex items-center justify-center w-14 h-14 mb-6 rounded-2xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 text-white shadow-xl shadow-indigo-500/10">
        <Sparkles className="w-7 h-7" />
      </div>

      {/* Main Heading */}
      <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
        How can I help you today?
      </h2>

      {/* Supporting text */}
      <p className="mt-3 text-sm sm:text-base text-muted-foreground max-w-lg leading-relaxed">
        Ask anything. Explore ideas, understand complex topics, and discover verified Wikipedia knowledge with Nexia.
      </p>

      {/* Suggested prompts grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full mt-8 text-left">
        {SUGGESTED_PROMPTS.map((item, idx) => {
          const Icon = item.icon
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              type="button"
              className="flex flex-col p-4 rounded-2xl border border-border/80 bg-card/60 hover:bg-card hover:border-border text-foreground transition-all duration-200 shadow-xs hover:shadow-md group text-left"
            >
              <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-muted-foreground group-hover:text-primary transition-colors">
                <Icon className="w-4 h-4 text-indigo-400" />
                <span>{item.title}</span>
              </div>
              <p className="text-xs text-foreground/80 line-clamp-2 leading-relaxed">
                {item.prompt}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

