const ANIMALS = ['dog', 'cat'];
const CLASSES = {
  dog: ['bark', 'growl', 'grunt'],
  cat: ['brushing', 'food', 'isolation'],
};
const TARGET_LEN = { dog: 64000, cat: 32000 };
const THRESHOLD = 0.5;

let mobilenet = null;
let headModels = {};

export async function loadModel() {
  const tf = await import('@tensorflow/tfjs');

  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch (e) {
    console.warn('WebGL backend failed, falling back to CPU:', e);
    await tf.setBackend('cpu');
    await tf.ready();
  }

  const mobilenetModule = await import('@tensorflow-models/mobilenet');

  mobilenet = await mobilenetModule.load({ version: 2, alpha: 1.0 });
  console.log('MobileNetV2 backbone loaded');

  for (const animal of ANIMALS) {
    try {
      headModels[animal] = await loadHeadWeights(animal, tf);
      console.log(`${animal} head weights loaded`);
    } catch (e) {
      console.warn(`${animal} head weights not loaded, predictions will throw:`, e);
    }
  }
}

async function loadHeadWeights(animal, tf) {
  const shapesRes = await fetch(`/model/${animal}/head_shapes.json`);
  const shapes = await shapesRes.json();

  const weightsRes = await fetch(`/model/${animal}/head_weights.bin`);
  const weightsBuf = await weightsRes.arrayBuffer();
  const weights = new Float32Array(weightsBuf);

  let offset = 0;
  const params = {};
  for (const [name, shape] of Object.entries(shapes)) {
    const size = shape.reduce((a, b) => a * b, 1);
    params[name] = tf.tensor(weights.slice(offset, offset + size), shape);
    offset += size;
  }

  return {
    dense_kernel: params.dense_kernel,
    dense_bias: params.dense_bias,
    dense_1_kernel: params.dense_1_kernel,
    dense_1_bias: params.dense_1_bias,
  };
}

function applyHead(embedding, head, tf) {
  return tf.tidy(() => {
    const h = tf.relu(tf.add(tf.matMul(embedding, head.dense_kernel), head.dense_bias));
    return tf.add(tf.matMul(h, head.dense_1_kernel), head.dense_1_bias);
  });
}

export async function classifyAudio(audioBlob, animal) {
  if (!ANIMALS.includes(animal)) throw new Error(`Unknown animal: ${animal}`);

  const tf = await import('@tensorflow/tfjs');

  if (!mobilenet || !headModels[animal]) {
    throw new Error(`Model not loaded for "${animal}". Call loadModel() first.`);
  }

  const imgTensor = await preprocessAudio(audioBlob, animal, tf);
  const embedding = mobilenet.infer(imgTensor, true);
  const logits = applyHead(embedding, headModels[animal], tf);
  const probsTensor = tf.softmax(logits);
  const probs = await probsTensor.data();

  tf.dispose([imgTensor, embedding, logits, probsTensor]);

  const classes = CLASSES[animal];
  const probsArray = Array.from(probs);
  const maxProb = Math.max(...probsArray);
  const topIdx = probsArray.indexOf(maxProb);
  const confidence = Math.round(maxProb * 10000) / 10000;
  const label = confidence >= THRESHOLD ? classes[topIdx] : 'uncertain';

  return {
    animal,
    label,
    confidence,
    probabilities: Object.fromEntries(classes.map((c, i) => [c, Math.round(probsArray[i] * 10000) / 10000])),
    threshold: THRESHOLD,
  };
}

function resampleLinear(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio;
  const ratio = toRate / fromRate;
  const out = new Float32Array(Math.round(audio.length * ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    out[i] = idx + 1 < audio.length
      ? audio[idx] * (1 - frac) + audio[idx + 1] * frac
      : audio[idx];
  }
  return out;
}

function fixLength(audio, targetLen) {
  if (audio.length >= targetLen) {
    const start = Math.floor((audio.length - targetLen) / 2);
    return audio.slice(start, start + targetLen);
  }
  const out = new Float32Array(targetLen);
  const padLeft = Math.floor((targetLen - audio.length) / 2);
  out.set(audio, padLeft);
  return out;
}

function hannWindow(size) {
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
}

function melSpectrogram(audio, sampleRate) {
  const nFft = 1024;
  const hopLength = 512;
  const nMels = 64;
  const fMin = 0;
  const fMax = sampleRate / 2;

  const numFrames = Math.floor((audio.length - nFft) / hopLength) + 1;
  const window = hannWindow(nFft);

  const fftBins = nFft / 2 + 1;
  const magSpectrogram = new Float32Array(numFrames * fftBins);

  for (let t = 0; t < numFrames; t++) {
    const offset = t * hopLength;
    for (let k = 0; k < fftBins; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < nFft; n++) {
        const angle = (2 * Math.PI * k * n) / nFft;
        const val = audio[offset + n] * window[n];
        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }
      magSpectrogram[t * fftBins + k] = re * re + im * im;
    }
  }

  const melBasis = createMelFilterBank(sampleRate, nFft, nMels, fMin, fMax);
  const melSpec = new Float32Array(numFrames * nMels);

  for (let t = 0; t < numFrames; t++) {
    for (let m = 0; m < nMels; m++) {
      let sum = 0;
      for (let k = 0; k < fftBins; k++) {
        sum += magSpectrogram[t * fftBins + k] * melBasis[m * fftBins + k];
      }
      melSpec[t * nMels + m] = 10 * Math.log10(sum + 1e-10);
    }
  }

  const nFrames = numFrames;
  const transposed = new Float32Array(nMels * nFrames);
  for (let t = 0; t < nFrames; t++) {
    for (let m = 0; m < nMels; m++) {
      transposed[m * nFrames + t] = melSpec[t * nMels + m];
    }
  }

  return { mel: transposed, nFrames };
}

function createMelFilterBank(sampleRate, nFft, nMels, fMin, fMax) {
  const fftBins = nFft / 2 + 1;

  // Slaney mel scale — matches librosa.filters.mel(htk=False)
  const F_SP = 200.0 / 3.0;
  const MIN_LOG_HZ = 1000.0;
  const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
  const LOGSTEP = Math.log(6.4) / 27.0;
  const hzToMel = f => f < MIN_LOG_HZ ? f / F_SP : MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOGSTEP;
  const melToHz = m => m < MIN_LOG_MEL ? m * F_SP : MIN_LOG_HZ * Math.exp(LOGSTEP * (m - MIN_LOG_MEL));

  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);

  const hzPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    hzPoints[i] = melToHz(melMin + (melMax - melMin) * i / (nMels + 1));
  }

  const bin = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    bin[i] = Math.floor((nFft + 1) * hzPoints[i] / sampleRate);
  }

  const basis = new Float32Array(nMels * fftBins);
  for (let m = 0; m < nMels; m++) {
    const fLeft = bin[m];
    const fCenter = bin[m + 1];
    const fRight = bin[m + 2];
    for (let k = fLeft; k <= fCenter; k++) {
      basis[m * fftBins + k] = (k - fLeft) / (fCenter - fLeft + 1e-10);
    }
    for (let k = fCenter; k <= fRight; k++) {
      basis[m * fftBins + k] = (fRight - k) / (fRight - fCenter + 1e-10);
    }
  }
  return basis;
}

async function preprocessAudio(audioBlob, animal, tf) {
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  let samples = audioBuffer.getChannelData(0);
  samples = resampleLinear(samples, audioBuffer.sampleRate, 16000);
  samples = fixLength(samples, TARGET_LEN[animal]);

  const { mel, nFrames } = melSpectrogram(samples, 16000);
  const nMels = 64;

  let minVal = Infinity, maxVal = -Infinity;
  for (const v of mel) {
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = maxVal - minVal + 1e-8;
  const scaled = mel.map(v => (v - minVal) / range);

  const rgb = new Float32Array(nMels * nFrames * 3);
  for (let i = 0; i < nMels * nFrames; i++) {
    rgb[i * 3] = scaled[i];
    rgb[i * 3 + 1] = scaled[i];
    rgb[i * 3 + 2] = scaled[i];
  }

  let imgTensor = tf.tensor3d(rgb, [nMels, nFrames, 3]);
  imgTensor = tf.image.resizeBilinear(imgTensor.expandDims(0), [96, 96]);
  imgTensor = imgTensor.mul(255.0);
  imgTensor = imgTensor.squeeze([0]);

  return imgTensor;
}

