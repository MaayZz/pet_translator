import { useState, useCallback } from "react";
import PetSelector from "./components/PetSelector";
import AudioRecorder from "./components/AudioRecorder";
import ChatUI from "./components/ChatUI";
import { classifyAudio } from "./lib/modelLoader";
import { translate } from "./lib/api";
import "./App.css";

function App() {
  const [petType, setPetType] = useState("cat");
  const [personality, setPersonality] = useState("snobbish");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

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
      { role: "pet", text: response.text, emotion: response.emotion, confidence: response.confidence, timestamp: response.timestamp || new Date().toISOString() },
    ]);
    setLoading(false);
  }, [petType, personality, messages]);

  return (
    <div className="desktop">
      <div className="phone-frame">
        <div className="phone-notch">
          <div className="notch-dynamic-island"></div>
        </div>
        <div className="phone-screen">
          <header className="app-header">
            <span className="header-pet">{petType === "cat" ? "🐱" : "🐶"}</span>
            <div className="header-info">
              <h1>Pet Translator</h1>
              <span className="header-subtitle">Mon{" "}{petType === "cat" ? "Chat" : "Chien"}</span>
            </div>
            <span className="header-battery">🔋</span>
          </header>

          <div className="phone-content">
            <ChatUI messages={messages} petType={petType} />
          </div>

          <div className="phone-controls">
            {loading && <div className="loading-bar">Traduction en cours...</div>}
            <div className="controls-row">
              <div className="controls-pet">
                <PetSelector
                  petType={petType}
                  personality={personality}
                  onPetChange={(p) => { setPetType(p); if (p === "cat") setPersonality("snobbish"); else setPersonality("excited"); }}
                  onPersonalityChange={setPersonality}
                />
              </div>
              <div className="controls-record">
                <AudioRecorder onAudioCaptured={handleAudioCaptured} />
              </div>
            </div>
          </div>
        </div>
        <div className="phone-home-bar"></div>
      </div>
    </div>
  );
}

export default App;
