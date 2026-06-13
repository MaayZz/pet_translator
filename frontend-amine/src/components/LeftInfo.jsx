const STACK = [
  { name: "Audio Processing", tech: "librosa / denoise / VAD", desc: "Noise reduction, voice activity detection, MFCC & mel-spectrogram extraction" },
  { name: "Classification", tech: "MobileNetV2 -> TFJS", desc: "Fine-tuned on pet vocalizations — 6 categories: hunger, play, attention, fear, pain, content" },
  { name: "Large Language Model", tech: "Llama 3.2 1B (LoRA)", desc: "Fine-tuned via unsloth on 5K+ synthetic pet phrases. Runs on llama.cpp" },
  { name: "Backend API", tech: "FastAPI + llama.cpp", desc: "REST API for LLM inference, conversation history storage, CORS-enabled" },
  { name: "Frontend", tech: "React + Vite + TFJS", desc: "In-browser audio capture, real-time classification via TensorFlow.js" },
];

export default function LeftInfo() {
  return (
    <aside className="side-panel left">
      <div className="panel-header">
        <h1>Pet Translation Device</h1>
        <div className="badge">ML01 — Fun & Experimental</div>
        <p className="desc">
          A collar-worn device that captures pet sounds, classifies their intent, and translates them into natural language via a fine-tuned LLM.
        </p>
      </div>

      <div className="panel-section">
        <h2>Tech Stack</h2>
        <div className="stack-list">
          {STACK.map((s, i) => (
            <div key={i} className="stack-item">
              <strong>{s.name}</strong>
              <span className="tech-code">{s.tech}</span>
              <p>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
