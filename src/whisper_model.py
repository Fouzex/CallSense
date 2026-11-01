from pathlib import Path
import torch
import whisper

_model = None


def load_whisper():
    global _model

    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = whisper.load_model("small", device=device)

    return _model


def transcribe_audio(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    model = load_whisper()

    result = model.transcribe(
        str(audio_path),
        language="ar",
        word_timestamps=True,
        fp16=torch.cuda.is_available(),
        verbose=False
    )

    words = []

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            words.append({
                "start_sec": round(float(word["start"]), 3),
                "end_sec": round(float(word["end"]), 3),
                "word": word["word"].strip()
            })

    return {
        "transcript": result["text"].strip(),
        "word_timestamps": words,
        "whisper_model": "small"
    }