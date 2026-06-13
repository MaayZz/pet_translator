import { useState, useRef } from "react";

export default function AudioRecorder({ onAudioCaptured }) {
  const [state, setState] = useState("idle");
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);

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

  return (
    <div className="audio-recorder">
      <button
        className={`record-btn ${state === "recording" ? "recording" : ""}`}
        onClick={state === "recording" ? stopRecording : startRecording}
        disabled={state === "processing"}
      >
        {state === "idle" && "🎤 Enregistrer"}
        {state === "recording" && "⏹ Arrêter"}
        {state === "processing" && "⏳ Analyse..."}
      </button>
      {state === "recording" && <div className="recording-indicator">Enregistrement en cours...</div>}
    </div>
  );
}
