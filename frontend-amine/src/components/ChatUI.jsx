import { useEffect, useRef } from "react";

const PET_EMOJIS = { cat: "🐱", dog: "🐶" };
const EMOTION_EMOJIS = {
  hunger: "🍽️",
  play: "🎾",
  attention: "👀",
  fear: "😰",
  pain: "💔",
  content: "😊",
};

export default function ChatUI({ messages, petType }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        {PET_EMOJIS[petType] || "🐾"} Discussion avec mon animal
      </div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            Appuyez sur "Enregistrer" pour commencer la discussion
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === "pet" && (
              <div className="message-avatar">{PET_EMOJIS[petType] || "🐾"}</div>
            )}
            <div className="message-bubble">
              <div className="message-text">{msg.text}</div>
              <div className="message-footer">
                <span className="message-emotion">
                  {EMOTION_EMOJIS[msg.emotion]} {msg.emotion}
                </span>
                <span className="message-time">
                  {new Date(msg.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
