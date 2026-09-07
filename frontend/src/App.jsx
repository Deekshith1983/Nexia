import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, 
  BookOpen, 
  Send, 
  Plus, 
  MessageSquare, 
  ShieldCheck, 
  ExternalLink,
  Cpu,
  RefreshCw
} from 'lucide-react';
import './App.css';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    // Check backend health status
    fetch('http://127.0.0.1:8000/api/health')
      .then((res) => res.json())
      .then((data) => setBackendStatus(data))
      .catch(() => setBackendStatus({ status: 'offline' }));
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (queryText) => {
    const query = queryText || input;
    if (!query.trim() || loading) return;

    const userMessage = { id: Date.now(), sender: 'user', text: query };
    setMessages((prev) => [...prev, userMessage]);
    if (!queryText) setInput('');
    setLoading(true);

    // Initial placeholder answer demonstrating grounded agent flow
    setTimeout(() => {
      const nexiaResponse = {
        id: Date.now() + 1,
        sender: 'nexia',
        text: `Nexia V1 Grounded Agent core initialized. In Stage 5 & 6, this prompt will be sent to Qwen3 8B via Ollama and verified against retrieved Wikipedia evidence.`,
        sources: [
          { title: 'Wikipedia Knowledge Engine', url: 'https://en.wikipedia.org' }
        ]
      };
      setMessages((prev) => [...prev, nexiaResponse]);
      setLoading(false);
    }, 800);
  };

  const samplePrompts = [
    "Why did the Roman Empire fall?",
    "Explain quantum mechanics like I'm 15",
    "Who was Nikola Tesla?",
    "What caused World War II?"
  ];

  return (
    <div className="nexia-container">
      {/* Sidebar */}
      <aside className="nexia-sidebar">
        <div className="sidebar-header">
          <div className="brand-logo">
            <div className="brand-icon">
              <Sparkles size={18} />
            </div>
            <span>Nexia</span>
          </div>
        </div>

        <button className="new-chat-btn" onClick={() => setMessages([])}>
          <Plus size={16} />
          New Conversation
        </button>

        <div className="chat-history-list">
          <div className="history-item active">
            <MessageSquare size={14} />
            <span>Current Session</span>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="status-badge">
            <div className="dot-pulse"></div>
            <span>Wikipedia Grounded</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem' }}>
            <Cpu size={12} color="#94a3b8" />
            <span>Qwen3 8B via Ollama</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Panel */}
      <main className="nexia-main">
        {/* Top Header */}
        <header className="top-bar">
          <div className="top-title">
            <span>Nexia Agent</span>
            <span className="model-pill">Qwen3 8B</span>
          </div>
          <div className="status-badge">
            <ShieldCheck size={14} />
            <span>
              Backend API: {backendStatus?.status === 'online' ? 'Online' : 'Checking...'}
            </span>
          </div>
        </header>

        {/* Chat Body */}
        <div className="chat-body">
          {messages.length === 0 ? (
            <div className="welcome-hero animate-fade-in">
              <div className="hero-icon">
                <BookOpen size={32} />
              </div>
              <h1 className="hero-title">Welcome to Nexia</h1>
              <p className="hero-subtitle">
                An AI agent designed to converse naturally, with every factual claim strictly grounded in Wikipedia.
              </p>

              <div className="sample-prompts">
                {samplePrompts.map((prompt, idx) => (
                  <button 
                    key={idx} 
                    className="prompt-card"
                    onClick={() => handleSend(prompt)}
                  >
                    "{prompt}"
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-container">
              {messages.map((msg) => (
                <div key={msg.id} className="chat-bubble animate-fade-in">
                  <div className={`avatar ${msg.sender}`}>
                    {msg.sender === 'user' ? 'U' : <Sparkles size={18} />}
                  </div>
                  <div className="message-content">
                    <p>{msg.text}</p>
                    {msg.sources && (
                      <div className="source-citations">
                        <span>Sources:</span>
                        {msg.sources.map((src, i) => (
                          <a 
                            key={i} 
                            href={src.url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="citation-chip"
                          >
                            <BookOpen size={12} />
                            {src.title}
                            <ExternalLink size={10} />
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="chat-bubble animate-fade-in">
                  <div className="avatar nexia">
                    <RefreshCw size={16} className="spin-icon" />
                  </div>
                  <div className="message-content" style={{ color: 'var(--text-muted)' }}>
                    Nexia is retrieving evidence from Wikipedia & reasoning...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="input-area-wrapper">
          <div className="input-box">
            <textarea
              className="chat-input"
              rows="1"
              placeholder="Ask Nexia anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <div className="input-controls">
              <div className="input-hint">
                <BookOpen size={12} color="#10b981" />
                <span>Powered by Wikipedia knowledge</span>
              </div>
              <button 
                className="send-btn" 
                disabled={!input.trim() || loading}
                onClick={() => handleSend()}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
