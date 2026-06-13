const PIPELINE = [
  { label: "Audio Capture", desc: "MediaRecorder API" },
  { label: "Classification", desc: "TF.js MobileNetV2" },
  { label: "LLM Inference", desc: "Llama GGUF server" },
  { label: "Chat Display", desc: "iMessage-style UI" },
];

const TEAM = [
  { name: "Mohamed", role: "Audio Pipeline & VAD" },
  { name: "Anas", role: "Classification Model" },
  { name: "Abir", role: "LLM Fine-tuning" },
  { name: "Amine", role: "Frontend & Integration" },
];

const EXPERIMENTS = [
  "Edge vs Cloud classification latency",
  "Before/After LoRA fine-tuning quality",
  "False positive rate on ambient noise",
  "Model quantization: FP32 vs INT8",
  "End-to-end latency measurement",
];

export default function RightInfo() {
  return (
    <aside className="side-panel right">
      <div className="panel-section">
        <h2>Pipeline</h2>
        <div className="pipeline">
          {PIPELINE.map((p, i) => (
            <div key={i} className="pipeline-step">
              <div className="step-dot" />
              <strong>{p.label}</strong>
              <span>{p.desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-section">
        <h2>Team</h2>
        <div className="team">
          {TEAM.map((m, i) => (
            <div key={i} className="team-member">
              <strong>{m.name}</strong>
              <span>{m.role}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel-section">
        <h2>Experiments</h2>
        <ul className="exp-list">
          {EXPERIMENTS.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
