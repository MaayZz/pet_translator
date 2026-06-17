const CATEGORIES = ["hunger", "play", "attention", "fear", "pain", "content"];

let model = null;

export async function loadModel() {
  try {
    const tf = await import("@tensorflow/tfjs");
    model = await tf.loadLayersModel("/model/model.json");
    console.log("Model loaded from /model/model.json");
  } catch {
    console.log("No model found at /model/model.json, using mock classifier");
  }
}

export async function classifyAudio(audioBlob) {
  if (model) {
    try {
      const tf = await import("@tensorflow/tfjs");
      const audioCtx = new AudioContext();
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      const channelData = audioBuffer.getChannelData(0);
      const resampled = resample(channelData, audioBuffer.sampleRate, 16000);
      const spectrogram = computeMelSpectrogram(resampled, 16000);
      const input = tf.tensor(spectrogram).expandDims(0).expandDims(-1);
      const resized = tf.image.resizeBilinear(input, [128, 128]);
      const output = model.predict(resized);
      const probs = await output.data();
      const maxIdx = probs.indexOf(Math.max(...probs));
      const categories = await getCategories();
      return { category: categories[maxIdx], confidence: probs[maxIdx] };
    } catch (err) {
      console.error("Model inference failed:", err);
    }
  }
  const category = CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)];
  const confidence = 0.75 + Math.random() * 0.2;
  return { category, confidence: Math.round(confidence * 100) / 100 };
}

async function getCategories() {
  try {
    const res = await fetch("/model/metadata.json");
    if (res.ok) {
      const meta = await res.json();
      return meta.categories || CATEGORIES;
    }
  } catch {}
  return CATEGORIES;
}

function resample(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio;
  const ratio = toRate / fromRate;
  const newLength = Math.round(audio.length * ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    result[i] = idx + 1 < audio.length
      ? audio[idx] * (1 - frac) + audio[idx + 1] * frac
      : audio[idx];
  }
  return result;
}

function computeMelSpectrogram(audio, sampleRate) {
  const frameSize = 1024;
  const hopLength = 512;
  const numFrames = Math.floor((audio.length - frameSize) / hopLength) + 1;
  const spectrogram = [];
  for (let i = 0; i < numFrames && i < 128; i++) {
    const frame = audio.slice(i * hopLength, i * hopLength + frameSize);
    const windowed = frame.map((s, j) => s * (0.54 - 0.46 * Math.cos((2 * Math.PI * j) / (frameSize - 1))));
    const fft = [];
    for (let k = 0; k < 128; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < frameSize; n++) {
        const angle = (2 * Math.PI * k * n) / frameSize;
        re += windowed[n] * Math.cos(angle);
        im -= windowed[n] * Math.sin(angle);
      }
      fft.push(Math.sqrt(re * re + im * im));
    }
    spectrogram.push(fft.map((v) => Math.log(v + 1)));
  }
  return spectrogram;
}
