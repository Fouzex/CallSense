"""Reusable Arabic TTS for CallSense."""

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


class ArabicTTS:
    def __init__(self, model_name="facebook/mms-tts-ara"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sample_rate = int(self.model.config.sampling_rate)

    @staticmethod
    def split_text(text, max_chars=180):
        text = re.sub(r"\[S\d+\]", "", str(text))
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            raise ValueError("TTS text cannot be empty")

        sentences = re.split(r"(?<=[.!؟?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            # Keep each chunk reasonably short
            words = sentence.split()
            for word in words:
                if len(current) + len(word) + 1 > max_chars:
                    if current:
                        chunks.append(current)
                    current = word
                else:
                    current = (current + " " + word).strip()

        if current:
            chunks.append(current)

        return chunks

    def synthesize(self, text, output_path="digest.wav"):
        audio_parts = []

        for chunk in self.split_text(text):
            inputs = self.tokenizer(chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                waveform = self.model(**inputs).waveform.squeeze().cpu().numpy()

            audio_parts.append(waveform.astype(np.float32))
            audio_parts.append(
                np.zeros(int(self.sample_rate * 0.25), dtype=np.float32)
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, np.concatenate(audio_parts), self.sample_rate)

        return str(output_path)
