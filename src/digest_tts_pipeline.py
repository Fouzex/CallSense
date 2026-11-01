"""Convert a generated CallSense digest to Arabic speech."""

from pathlib import Path
from src.tts_inference import ArabicTTS


class DigestTTSPipeline:
    def __init__(self):
        self.tts = ArabicTTS()

    def predict(self, digest_text, output_path="digest.wav"):
        if not isinstance(digest_text, str) or not digest_text.strip():
            raise ValueError("Digest text cannot be empty")

        audio_path = self.tts.synthesize(
            text=digest_text,
            output_path=output_path,
        )

        return {
            "digest_text": digest_text,
            "audio_path": str(Path(audio_path)),
        }
