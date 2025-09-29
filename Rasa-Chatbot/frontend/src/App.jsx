// Rasa-Chatbot/frontend/src/App.jsx

import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const chatEndRef = useRef(null)

  async function sendMessage() {
    if (!input.trim()) return

    const userMessage = { from: 'user', text: input }
    setMessages(prev => [...prev, userMessage])
    const messageText = input
    setInput('')

    try {
      const res = await fetch('http://localhost:8000/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sender: 'user1', message: messageText })
      })
      const data = await res.json()
      const botMessage = { from: 'bot', text: data.reply || '...' }
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      setMessages(prev => [...prev, { from: 'bot', text: 'Error: Could not connect to server' }])
    }
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="chat-container">
      <h1>✨ Hugging Face Chatbot ✨</h1>
      <div className="chat-window">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.from}`}>{m.text}</div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <div className="chat-input">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  )
}

export default App
