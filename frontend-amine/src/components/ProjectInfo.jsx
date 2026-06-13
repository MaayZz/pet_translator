const STACK = [
  { icon: "🎤", name: "Audio Processing", tech: "librosa · denoise · VAD", desc: "Noise reduction, voice activity detection, MFCC & mel-spectrogram extraction" },
  { icon: "🧠", name: "Classification", tech: "MobileNetV2 → TFJS", desc: "Fine-tuned on pet vocalizations — 6 categories: hunger, play, attention, fear, pain, content" },
  { icon: "🤖", name: "LLM", tech: "Llama 3.2 1B (LoRA)", desc: "Fine-tuned via unsloth on 5K+ synthetic pet phrases. Runs on llama.cpp backend" },
  { icon: "⚡", name: "Backend", tech: "FastAPI + llama.cpp", desc: "REST API for LLM inference, conversation history, CORS-enabled" },
  { icon: "📱", name: "Frontend", tech: "React + Vite + TFJS", desc: "In-browser audio capture, real-time classification via TensorFlow.js" },
];

const PIPELINE = [
  { label: "Audio Capture", icon: "🎙️", desc: "MediaRecorder API" },
  { label: "Classification", icon: "🧠", desc: "TF.js MobileNetV2" },
  { label: "LLM Inference", icon: "🤖", desc: "Llama GGUF (server)" },
  { label: "Chat Display", icon: "💬", desc: "iMessage-style UI" },
];

const TEAM = [
  { name: "Member 1", role: "Audio Pipeline & VAD", emoji: "🎵" },
  { name: "Member 2", role: "Classification Model", emoji: "🧠" },
  { name: "Member 3", role: "LLM Fine-tuning", emoji: "🤖" },
  { name: "Member 4", role: "Frontend & Integration", emoji: "📱" },
];

export default function ProjectInfo() {
  return (
    <aside className="project-info">
      <div className="info-header">
        <h1>Pet Translation Device</h1>
        <p className="info-badge">ML01 — Fun & Experimental</p>
        <p className="info-subtitle">
          A collar-worn device that captures pet sounds, classifies their intent, and translates them into natural language via a fine-tuned LLM.
        </p>
      </div>

      <div className="info-section">
        <h2>📋 Tech Stack</h2>
        <div className="stack-grid">
          {STACK.map((s, i) => (
            <div key={i} className="stack-card">
              <div className="stack-icon">{s.icon}</div>
              <div className="stack-body">
                <strong>{s.name}</strong>
                <code>{s.tech}</code>
                <p>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="info-section">
        <h2>🔁 Pipeline</h2>
        <div className="pipeline">
          {PIPELINE.map((p, i) => (
            <>
              <div key={i} className="pipeline-node">
                <div className="pipeline-icon">{p.icon}</div>
                <strong>{p.label}</strong>
                <span>{p.desc}</span>
              </div>
              {i < PIPELINE.length - 1 && <div className="pipeline-arrow">→</div>}
            </>
          ))}
        </div>
      </div>

      <div className="info-section">
        <h2>👥 Team</h2>
        <div className="team-list">
          {TEAM.map((m, i) => (
            <div key={i} className="team-card">
              <span className="team-emoji">{m.emoji}</span>
              <div>
                <strong>{m.name}</strong>
                <span>{m.role}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="info-section">
        <h2>🔬 Experiments</h2>
        <ul className="experiments">
          <li>Edge vs Cloud classification latency</li>
          <li>Before/After LoRA fine-tuning quality</li>
          <li>False positive rate on ambient noise</li>
          <li>Model quantization: FP32 vs INT8</li>
          <li>End-to-end latency measurement</li>
        </ul>
      </div>
    </aside>
  );
}
