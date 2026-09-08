import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Sparkles,
  BookOpen,
  Send,
  Square,
  Plus,
  MessageSquare,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import './App.css';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const chatEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  const fetchHealthStatus = () => {
    fetch('http://127.0.0.1:8000/api/health')
      .then((res) => res.json())
      .then((data) => setBackendStatus(data))
      .catch(() => setBackendStatus({ status: 'offline', ollama_reachable: false }));
  };

  useEffect(() => {
    fetchHealthStatus();
    const interval = setInterval(fetchHealthStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ──────────────────────────────────────────────
  // Stop Generation — abort active fetch request
  // ──────────────────────────────────────────────
  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  // ──────────────────────────────────────────────
  // Send message + consume streamed response
  // ──────────────────────────────────────────────
  const handleSend = async (queryText) => {
    const query = queryText || input;
    if (!query.trim() || loading) return;

    const userMessage = { id: Date.now(), sender: 'user', text: query };
    const nexiaMessageId = Date.now() + 1;
    const initialNexiaMessage = {
      id: nexiaMessageId,
      sender: 'nexia',
      text: '',
      model: 'gemma3:4b',
      stopped: false,
      isError: false,
    };

    setMessages((prev) => [...prev, userMessage, initialNexiaMessage]);
    if (!queryText) setInput('');
    setLoading(true);

    // Create a fresh AbortController for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, conversation_id: conversationId }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to connect to Nexia backend server.');
      }

      const activeCid = response.headers.get('X-Conversation-Id');
      if (activeCid && activeCid !== conversationId) {
        setConversationId(activeCid);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        accumulatedText += chunk;

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === nexiaMessageId
              ? { ...msg, text: accumulatedText }
              : msg
          )
        );
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // User intentionally stopped — preserve partial response, no error badge
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === nexiaMessageId
              ? { ...msg, stopped: true }
              : msg
          )
        );
      } else {
        // Genuine network / server error
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === nexiaMessageId
              ? { ...msg, isError: true, text: `⚠️ ${err.message}` }
              : msg
          )
        );
      }
    } finally {
      abortControllerRef.current = null;
      setLoading(false);
    }
  };

  const samplePrompts = [
    'What is the capital of India?',
    'If I have 10 apples and give away 3, how many remain?',
    'Write a Python function to reverse a string.',
    'Explain machine learning in detail.',
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

        <button
          className="new-chat-btn"
          onClick={() => {
            setMessages([]);
            setConversationId(null);
          }}
        >
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
            <span>Stage 2 — Gemma LLM</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.2rem' }}>
            <Cpu size={12} color="#94a3b8" />
            <span>gemma3:4b via Ollama</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Panel */}
      <main className="nexia-main">
        {/* Top Header */}
        <header className="top-bar">
          <div className="top-title">
            <span>Nexia Agent</span>
            <span className="model-pill">Gemma 3 4B</span>
          </div>
          <div className="status-badge">
            <ShieldCheck size={14} />
            <span>
              Backend: {backendStatus?.status === 'online' ? 'Online' : 'Offline'} |{' '}
              Ollama: {backendStatus?.ollama_reachable ? 'Ready' : 'Offline'}
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
                Stage 2: Powered by Gemma 3 4B inference through local Ollama service.
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
                    {msg.isError ? (
                      <p style={{ color: '#ef4444', fontWeight: 500 }}>{msg.text}</p>
                    ) : msg.sender === 'user' ? (
                      <p>{msg.text}</p>
                    ) : (
                      <>
                        {!msg.text && loading && (
                          <span className="thinking-cursor">▋</span>
                        )}
                        {msg.text && (
                          <div className="markdown-body">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {msg.text}
                            </ReactMarkdown>
                          </div>
                        )}
                        {msg.stopped && (
                          <p className="stopped-label">⏹ Generation stopped by user.</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
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
              placeholder="Ask Nexia (Gemma 3 4B)..."
              value={input}
              disabled={loading}
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
                <Cpu size={12} color="#3b82f6" />
                <span>Gemma 3 4B Local Model</span>
              </div>
              {loading ? (
                <button className="stop-btn" onClick={handleStop} title="Stop generation">
                  <Square size={14} />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  className="send-btn"
                  disabled={!input.trim()}
                  onClick={() => handleSend()}
                >
                  <Send size={16} />
                </button>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
