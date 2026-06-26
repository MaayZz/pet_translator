import { useState, useCallback, useEffect } from "react";
import PetSelector from "./components/PetSelector";
import AudioRecorder from "./components/AudioRecorder";
import ChatUI from "./components/ChatUI";
import LeftInfo from "./components/LeftInfo";
import RightInfo from "./components/RightInfo";
import { classifyAudio } from "./lib/modelLoader";
import { translate } from "./lib/api";
import "./App.css";

const KONAMI = ["ArrowUp","ArrowUp","ArrowDown","ArrowDown","ArrowLeft","ArrowRight","ArrowLeft","ArrowRight","b","a"];

function playBark() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(400, ctx.currentTime);
    osc.frequency.linearRampToValueAtTime(200, ctx.currentTime + 0.08);
    osc.frequency.linearRampToValueAtTime(300, ctx.currentTime + 0.12);
    osc.frequency.linearRampToValueAtTime(150, ctx.currentTime + 0.2);
    gain.gain.setValueAtTime(0.4, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.25);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.25);
    setTimeout(() => {
      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.type = "sawtooth";
      osc2.frequency.setValueAtTime(350, ctx.currentTime);
      osc2.frequency.linearRampToValueAtTime(180, ctx.currentTime + 0.1);
      gain2.gain.setValueAtTime(0.3, ctx.currentTime);
      gain2.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.2);
      osc2.start(ctx.currentTime);
      osc2.stop(ctx.currentTime + 0.2);
    }, 200);
  } catch { /* ignore */ }
}

function App() {
  const [petType, setPetType] = useState("cat");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isLight, setIsLight] = useState(false);
  const [showEasterEgg, setShowEasterEgg] = useState(false);

  useEffect(() => {
    let seq = [];
    const handler = (e) => {
      seq.push(e.key);
      if (seq.length > KONAMI.length) seq.shift();
      if (seq.length === KONAMI.length && seq.every((k, i) => k === KONAMI[i])) {
        seq = [];
        setShowEasterEgg(true);
        playBark();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleAudioCaptured = useCallback(async (audioBlob) => {
    setLoading(true);
    const result = await classifyAudio(audioBlob, petType);
    const response = await translate({
      animal: result.animal,
      label: result.label,
      confidence: result.confidence,
      probabilities: result.probabilities,
      history: messages,
    });
    setMessages((prev) => [
      ...prev,
      { role: "pet", text: response.text, emotion: response.emotion || result.label, confidence: response.confidence, timestamp: response.timestamp || new Date().toISOString(), petType: result.animal },
    ]);
    setLoading(false);
  }, [petType, messages]);

  return (
    <div className={"app-layout" + (isLight ? " light-theme" : "")}>
      <button className="theme-toggle" onClick={() => setIsLight(!isLight)} title="Toggle theme">
        {isLight ? "🌙" : "☀️"}
      </button>
      <LeftInfo />
      <div className="phone-section">
        <div className="phone-screen">
          <header className="app-header">
            <span className="header-avatar">{petType === "cat" ? "🐱" : "🐶"}</span>
            <span className="header-title">Pet Translator</span>
          </header>
          <div className="phone-content">
            <ChatUI messages={messages} />
          </div>
          <div className="phone-footer">
            {loading && <div className="loading-bar">Translating...</div>}
            <div className="footer-row">
              <PetSelector petType={petType} onPetChange={setPetType} />
              <AudioRecorder onAudioCaptured={handleAudioCaptured} />
            </div>
          </div>
        </div>
      </div>
      <RightInfo />
      {showEasterEgg && (
        <div className="easter-overlay" onClick={() => setShowEasterEgg(false)}>
          <div className="easter-content" onClick={(e) => e.stopPropagation()}>
            <img src="/easter-egg.jpeg" alt="Easter Egg" className="easter-image" />
            <p className="easter-label">WOOF! 🐶</p>
            <button className="easter-close" onClick={() => setShowEasterEgg(false)}>✕</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
