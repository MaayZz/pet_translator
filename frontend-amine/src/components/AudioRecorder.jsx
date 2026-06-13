import { useState, useRef } from "react";

export default function AudioRecorder({ onAudioCaptured }) {
  const [state, setState] = useState("idle");
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const fileInput = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunks.current = [];

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      mediaRecorder.current.onstop = async () => {
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        setState("processing");
        await onAudioCaptured(blob);
        setState("idle");
      };

      mediaRecorder.current.start();
      setState("recording");
    } catch (err) {
      console.error("Microphone access denied:", err);
      alert("Accès au microphone refusé. Autorisez-le dans votre navigateur.");
      setState("idle");
    }
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setState("processing");
    onAudioCaptured(file);
    setState("idle");
    e.target.value = "";
  };

  return (
    <div className="audio-recorder">
      <input
        ref={fileInput}
        type="file"
        accept="audio/*"
        onChange={handleFileUpload}
        style={{ display: "none" }}
      />
      <button
        className="record-btn"
        onClick={() => fileInput.current?.click()}
        title="Upload audio"
      >
        📂
      </button>
      <button
        className={`record-btn ${state === "recording" ? "recording" : ""}`}
        onClick={state === "recording" ? stopRecording : startRecording}
        disabled={state === "processing"}
        title={state === "idle" ? "Enregistrer" : state === "recording" ? "Arrêter" : "Analyse..."}
      >
        {state === "recording" ? "⏹" : "🎤"}
      </button>
    </div>
  );
}
