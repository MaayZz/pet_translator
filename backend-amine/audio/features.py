"""
Feature extraction for audio classification.

Functions:
    extract_mfcc(audio, sr) -> np.ndarray
    extract_mel_spectrogram(audio, sr) -> np.ndarray
"""

def extract_mfcc(audio, sr):
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
        return mfccs.T
    except Exception as e:
        raise RuntimeError(f"MFCC extraction failed: {e}")

def extract_mel_spectrogram(audio, sr):
    try:
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        log_mel = librosa.power_to_db(mel, ref=1.0)
        return log_mel.T
    except Exception as e:
        raise RuntimeError(f"Mel spectrogram extraction failed: {e}")
