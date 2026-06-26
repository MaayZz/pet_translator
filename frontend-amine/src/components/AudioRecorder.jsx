import { useState, useRef, useEffect } from "react";

export default function AudioRecorder({ onAudioCaptured }) {
  const [state, setState] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const fileInput = useRef(null);
  const startTime = useRef(0);

  useEffect(() => {
    if (errorMsg) {
      const timer = setTimeout(() => setErrorMsg(""), 3000);
      return () => clearTimeout(timer);
    }
  }, [errorMsg]);

  const startRecording = async () => {
    try {
      setErrorMsg("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunks.current = [];
      mediaRecorder.current.ondataavailable = (e) => { if (e.data.size > 0) chunks.current.push(e.data); };
      mediaRecorder.current.onstop = async () => {
        const duration = Date.now() - startTime.current;
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        stream.getTracks().forEach((t) => t.stop());
        
        if (duration < 1000) {
          setErrorMsg("Too short!");
          setState("idle");
          return;
        }

        setState("processing");
        await onAudioCaptured(blob);
        setState("idle");
      };
      mediaRecorder.current.start();
      startTime.current = Date.now();
      setState("recording");
    } catch (err) {
      setErrorMsg("Mic access denied");
      setState("idle");
    }
  };

  const stopRecording = () => mediaRecorder.current?.stop();

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setErrorMsg("");
    setState("processing");
    onAudioCaptured(file);
    setState("idle");
    e.target.value = "";
  };

  return (
    <div className="audio-recorder" style={{ position: "relative" }}>
      {errorMsg && (
        <div style={{
          position: "absolute", top: "-40px", left: "50%", transform: "translateX(-50%)",
          background: "var(--phone-danger)", color: "#fff", padding: "6px 12px",
          borderRadius: "8px", fontSize: "12px", fontWeight: "600", whiteSpace: "nowrap",
          boxShadow: "0 4px 12px rgba(255, 71, 87, 0.3)", animation: "easterPop 0.3s ease"
        }}>
          {errorMsg}
        </div>
      )}
      <input ref={fileInput} type="file" accept="audio/*" onChange={handleFileUpload} hidden />
      <button className="record-btn outline" onClick={() => fileInput.current?.click()} title="Upload audio file">📂</button>
      <button
        className={`record-btn ${state === "recording" ? "recording" : "solid"}`}
        onClick={state === "recording" ? stopRecording : startRecording}
        disabled={state === "processing"}
      >
        {state === "recording" ? "⏹" : "🎤"}
      </button>
    </div>
  );
}
