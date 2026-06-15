from pathlib import Path
import librosa
import noisereduce as nr
import soundfile as sf

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
TARGET_SR = 16000

def clean_audio_file(filepath, sr=TARGET_SR):
    y, sr = librosa.load(filepath, sr=sr)
    y_clean = nr.reduce_noise(y=y, sr=sr)
    return y_clean, sr

def process_dataset(raw_dir, clean_dir):
    raw_dir = Path(raw_dir)
    clean_dir = Path(clean_dir)

    wav_files = list(raw_dir.rglob("*.wav"))
    print(f"Nombre total de fichiers trouvés : {len(wav_files)}")

    failed_files = []

    for i, filepath in enumerate(wav_files):
        relative_path = filepath.relative_to(raw_dir)
        output_path = clean_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y_clean, sr = clean_audio_file(filepath)
            sf.write(output_path, y_clean, sr)
        except Exception as e:
            print(f"ERREUR sur {filepath} : {e}")
            failed_files.append(filepath)
            continue

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(wav_files)} fichiers traités")

    print(f"\nTerminé. {len(wav_files) - len(failed_files)} fichiers traités avec succès.")
    if failed_files:
        print(f"ATTENTION : {len(failed_files)} fichiers ont échoué :")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    process_dataset(RAW_DIR, CLEAN_DIR)