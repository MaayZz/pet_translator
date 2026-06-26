const RESEARCH_HIGHLIGHTS = [
  {
    name: "Dataset",
    tech: "350+ labeled pet vocalizations",
    desc: "Dog: 62 bark, 60 growl, 48 grunt — Cat: 85 brushing, 56 food, 48 isolation. Split train/val/test (70/15/15).",
  },
  {
    name: "Features",
    tech: "Log-mel spectrograms (64 bands)",
    desc: "16 kHz mono -> center crop (4s dog / 2s cat) -> Hann window (1024) -> 64 mel bands -> 10*log10(power) -> per-sample min-max -> 3-channel image -> resize 96x96.",
  },
  {
    name: "Backbone",
    tech: "MobileNetV2 (frozen, ImageNet)",
    desc: "GlobalAveragePooling -> 1280-d embedding. Fully convolutional — accepts 96x96 input. Same backbone for both species; only the head is per-animal.",
  },
  {
    name: "Classification Head",
    tech: "Dense(64, ReLU) -> Dense(3, Softmax)",
    desc: "Dog classes: bark, growl, grunt — Cat classes: brushing, food, isolation. Confidence threshold: 0.50 (below = uncertain).",
  },
  {
    name: "LLM",
    tech: "Llama 3.2 1B (LoRA, unsloth)",
    desc: "Fine-tuned on 5K+ synthetic pet phrases. 4 personalities (haughty_cat, grumpy_cat, excited_dog, shy_dog). Falls back to mock if GGUF unavailable.",
  },
  {
    name: "Deployment",
    tech: "TFJS in-browser + FastAPI",
    desc: "Backbone via @tensorflow-models/mobilenet; trained head weights extracted as .bin (1280->64->3). Inference runs entirely in the browser.",
  },
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
        <h2>Research Architecture</h2>
        <div className="stack-list">
          {RESEARCH_HIGHLIGHTS.map((s, i) => (
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
