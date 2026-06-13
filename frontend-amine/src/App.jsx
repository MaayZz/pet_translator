import { useState, useCallback } from "react";
import PetSelector from "./components/PetSelector";
import AudioRecorder from "./components/AudioRecorder";
import ChatUI from "./components/ChatUI";
import { classifyAudio } from "./lib/modelLoader";
import { translate } from "./lib/api";
import ProjectInfo from "./components/ProjectInfo";
import "./App.css";

function App() {
  const [petType, setPetType] = useState("cat");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleAudioCaptured = useCallback(async (audioBlob) => {
    setLoading(true);
    const result = await classifyAudio(audioBlob);
    const response = await translate({
      category: result.category,
      confidence: result.confidence,
      petType,
      history: messages,
    });
    setMessages((prev) => [
      ...prev,
      { role: "pet", text: response.text, emotion: response.emotion, confidence: response.confidence, timestamp: response.timestamp || new Date().toISOString(), petType },
    ]);
    setLoading(false);
  }, [petType, messages]);

  return (
    <div className="app-layout">
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
            {loading && <div className="loading-bar">✨ Translating...</div>}
            <div className="footer-row">
              <PetSelector petType={petType} onPetChange={setPetType} />
              <AudioRecorder onAudioCaptured={handleAudioCaptured} />
            </div>
          </div>
        </div>
      </div>
      <ProjectInfo />
    </div>
  );
}

export default App;
