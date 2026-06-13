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
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <span className="chat-empty-icon">{PET_EMOJIS[petType] || "🐾"}</span>
            <div className="chat-empty-title">Prêt à discuter</div>
            <div>Appuyez sur le bouton d'enregistrement pour traduire les sons de votre animal</div>
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
                  {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
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
