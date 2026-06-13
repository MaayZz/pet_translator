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
    <div className="app">
      <header className="app-header">
        <h1>🐾 Pet Translator</h1>
        <p className="subtitle">Traducteur animalier IA</p>
      </header>
      <main className="app-main">
        <aside className="sidebar">
          <PetSelector
            petType={petType}
            personality={personality}
            onPetChange={(p) => { setPetType(p); if (p === "cat") setPersonality("snobbish"); else setPersonality("excited"); }}
            onPersonalityChange={setPersonality}
          />
          <AudioRecorder onAudioCaptured={handleAudioCaptured} />
          {loading && <div className="loading-bar">Traduction en cours...</div>}
        </aside>
        <section className="chat-section">
          <ChatUI messages={messages} petType={petType} />
        </section>
      </main>
    </div>
  );
}

export default App;
