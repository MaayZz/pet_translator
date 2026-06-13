"""
Audio preprocessing pipeline for pet vocalizations.

Functions:
    load_audio(path) -> tuple[np.ndarray, int]
    reduce_noise(audio, sr) -> np.ndarray
    voice_activity_detection(audio, sr) -> list[tuple[int, int]]
"""

def load_audio(path):
    try:
        import librosa
        audio, sr = librosa.load(path, sr=16000, mono=True)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

def reduce_noise(audio, sr):
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
    except ImportError:
        return audio

def voice_activity_detection(audio, sr):
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        frame_size = int(sr * 0.03)
        segments = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            pcm = (frame * 32767).astype("int16").tobytes()
            if vad.is_speech(pcm, sr):
                segments.append((i, i + frame_size))
        return segments
    except ImportError:
        return [(0, len(audio))]
