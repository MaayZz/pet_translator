import { useEffect, useRef } from "react";

const PET_EMOJIS = { cat: "🐱", dog: "🐶" };
const EMOTION_EMOJIS = {
  bark: "🔊", growl: "😠", grunt: "😤",
  brushing: "🪥", food: "🍽️", isolation: "😿",
  uncertain: "🤔", no_sound: "🔇",
};

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function ProbBar({ label, prob, maxProb, isTop }) {
  const pct = (prob * 100).toFixed(1);
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-bar-track">
        <div
          className={"prob-bar-fill" + (isTop ? " prob-bar-top" : "")}
          style={{ width: pct + "%" }}
        />
      </div>
      <span className={"prob-value" + (isTop ? " prob-value-top" : "")}>{pct}%</span>
    </div>
  );
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
        {messages.map((msg, i) => {
          const probs = msg.probabilities || {};
          const entries = Object.entries(probs);
          const maxProb = entries.length > 0 ? Math.max(...Object.values(probs), 0) : 0;

          return (
            <div key={i} className="message">
              <div className="message-avatar">{PET_EMOJIS[msg.petType] || "🐾"}</div>
              <div className="message-bubble">
                <div className="message-text">{msg.text}</div>
                {entries.length > 0 && (
                  <div className="message-probs">
                    {entries.map(([name, prob]) => (
                      <ProbBar key={name} label={name} prob={prob} maxProb={maxProb} isTop={prob === maxProb} />
                    ))}
                  </div>
                )}
                <div className="message-footer">
                  <span>{EMOTION_EMOJIS[msg.emotion] || "🐾"} {capitalize(msg.emotion)}</span>
                  <span className="confidence-badge">{(msg.confidence * 100).toFixed(0)}%</span>
                  <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
