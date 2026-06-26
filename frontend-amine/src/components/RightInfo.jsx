const PIPELINE = [
  { label: "Audio In", desc: "MediaRecorder (webm) / File upload (wav)" },
  { label: "Resample", desc: "16 kHz mono via AudioContext" },
  { label: "Center Crop", desc: "64 000 samples (dog) / 32 000 (cat)" },
  { label: "Spectrogram", desc: "Hann(1024) -> 64 mel bands -> dB -> min-max -> 3ch -> 96x96" },
  { label: "Backbone", desc: "MobileNetV2 (frozen) -> 1280-d embedding" },
  { label: "Head", desc: "Dense(64,ReLU) -> Dense(3,Softmax) -> class probs" },
  { label: "LLM", desc: "Llama 3.2 1B (LoRA) -> natural language output" },
];

const TEAM = [
  { name: "Mohamed", role: "Audio Pipeline (denoise, VAD, librosa)" },
  { name: "Anas", role: "Classification (MobileNetV2 + head, TF/TFJS)" },
  { name: "Abir", role: "LLM (LoRA fine-tune, RAG, prompt engineering)" },
  { name: "Amine", role: "Frontend + Intégration (React, TFJS, FastAPI)" },
];

const EXPERIMENTS = [
  "Dog vs cat: shared backbone + per-animal head (vs separate full models)",
  "n_mels sweep (32 / 64 / 128) -> 64 chosen (accuracy vs speed)",
  "Input size sweep (64x64 / 96x96 / 128x128) -> 96x96 optimal",
  "Confidence threshold calibration (0.50 minimizes false positives)",
  "Ablation: with/without global norm before min-max (no effect, removed)",
];

const ACCURACY = {
  dog: { classes: ["bark", "growl", "grunt"], test_size: 12, accuracy: "92% (11/12)" },
  cat: { classes: ["brushing", "food", "isolation"], test_size: 15, accuracy: "87% (13/15)" },
};

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
        <h2>Test Accuracy</h2>
        <div className="accuracy-grid">
          {Object.entries(ACCURACY).map(([animal, data]) => (
            <div key={animal} className="accuracy-card">
              <strong>{animal.charAt(0).toUpperCase() + animal.slice(1)}</strong>
              <span className="acc-value">{data.accuracy}</span>
              <span className="acc-detail">n={data.test_size} | classes: {data.classes.join(", ")}</span>
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
