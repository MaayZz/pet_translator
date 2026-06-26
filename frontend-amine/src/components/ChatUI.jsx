import { useEffect, useRef } from "react";

const PET_EMOJIS = { cat: "🐱", dog: "🐶" };
const EMOTION_EMOJIS = {
  bark: "🔊", growl: "😠", grunt: "😤",
  brushing: "🪥", food: "🍽️", isolation: "😿",
  uncertain: "🤔",
};

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export default function ChatUI({ messages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <span className="chat-empty-icon">🐾</span>
            <div className="chat-empty-title">Ready to chat</div>
            <div>Press the record button to translate your pet's sounds</div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className="message">
            <div className="message-avatar">{PET_EMOJIS[msg.petType] || "🐾"}</div>
            <div className="message-bubble">
              <div className="message-text">{msg.text}</div>
              <div className="message-footer">
                <span>{EMOTION_EMOJIS[msg.emotion] || "🐾"} {capitalize(msg.emotion)}</span>
                <span className="confidence-badge">{(msg.confidence * 100).toFixed(0)}%</span>
                <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
              </div>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
