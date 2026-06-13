import { useState, useCallback } from "react";
import PetSelector from "./components/PetSelector";
import AudioRecorder from "./components/AudioRecorder";
import ChatUI from "./components/ChatUI";
import { classifyAudio } from "./lib/modelLoader";
import { translate } from "./lib/api";
import "./App.css";

const PERSONALITY_LABELS = {
  snobbish: "Hautain", timid: "Timide", grumpy: "Grumpy",
  excited: "Surexcité", playful: "Joueur", gentle: "Doux",
};

const CAT_PERSONALITIES = ["snobbish", "timid", "grumpy"];
const DOG_PERSONALITIES = ["excited", "playful", "gentle"];

function App() {
  const [petType, setPetType] = useState("cat");
  const [personality, setPersonality] = useState("snobbish");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handlePetChange = (p) => {
    setPetType(p);
    setPersonality(p === "cat" ? "snobbish" : "excited");
  };

  const handleAudioCaptured = useCallback(async (audioBlob) => {
    setLoading(true);
    const result = await classifyAudio(audioBlob);
    const response = await translate({
      category: result.category,
      confidence: result.confidence,
      petType,
      personality,
      history: messages,
    });
    setMessages((prev) => [
      ...prev,
      { role: "pet", text: response.text, emotion: response.emotion, confidence: response.confidence, timestamp: response.timestamp || new Date().toISOString(), petType },
    ]);
    setLoading(false);
  }, [petType, personality, messages]);

  return (
    <div className="desktop">
      <div className="phone-screen">
        <header className="app-header">
          <div className="header-avatar">
            {petType === "cat" ? "🐱" : "🐶"}
          </div>
          <div className="header-info">
            <h1>Pet Translator</h1>
            <select
              value={personality}
              onChange={(e) => setPersonality(e.target.value)}
              className="personality-select"
            >
              {(petType === "cat" ? CAT_PERSONALITIES : DOG_PERSONALITIES).map((p) => (
                <option key={p} value={p}>{PERSONALITY_LABELS[p]}</option>
              ))}
            </select>
          </div>
        </header>

        <div className="phone-content">
          <ChatUI messages={messages} />
        </div>

        <div className="phone-controls">
          {loading && <div className="loading-bar">✨ Traduction en cours...</div>}
          <div className="controls-row">
            <PetSelector petType={petType} onPetChange={handlePetChange} />
            <AudioRecorder onAudioCaptured={handleAudioCaptured} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
