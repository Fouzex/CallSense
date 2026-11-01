"""Audio -> Whisper -> Diarization -> Customer intent inference."""

import os
from pathlib import Path

from src.whisper_model import transcribe_audio
from src.diarization import SpeakerDiarizer
from src.windowed_intent_inference import WindowedIntentPipeline


class AudioIntentPipeline:
    def __init__(self, hf_token=None):
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required for speaker diarization")

        self.diarizer = SpeakerDiarizer(hf_token=token)
        self.intent_model = WindowedIntentPipeline()

    def predict(
        self,
        audio_path,
        customer_speaker,
        call_id=None,
        strategy="max_pooling",
    ):
        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)

        if customer_speaker not in {"SPEAKER_00", "SPEAKER_01"}:
            raise ValueError(
                "customer_speaker must be SPEAKER_00 or SPEAKER_01"
            )

        call_id = call_id or audio_path.stem

        # 1. Speech-to-text with word timestamps
        whisper_result = transcribe_audio(str(audio_path))
        words = whisper_result["word_timestamps"]

        # 2. Speaker diarization
        segments = self.diarizer.process(
            str(audio_path),
            num_speakers=2,
        )

        # 3. Assign words to speakers using timestamp midpoint
        aligned_words = []

        for word in words:
            midpoint = (
                word["start_sec"] + word["end_sec"]
            ) / 2

            speaker = next(
                (
                    segment["speaker"]
                    for segment in segments
                    if segment["start"] <= midpoint <= segment["end"]
                ),
                "UNKNOWN",
            )

            aligned_words.append({
                "start_sec": word["start_sec"],
                "end_sec": word["end_sec"],
                "word": word["word"],
                "speaker": speaker,
            })

        # 4. Extract customer speech
        customer_words = [
            item["word"]
            for item in aligned_words
            if item["speaker"] == customer_speaker
        ]

        customer_transcript = " ".join(customer_words).strip()

        if not customer_transcript:
            raise ValueError(
                "No customer speech found for selected speaker"
            )

        # 5. Predict customer intents
        intent_result = self.intent_model.predict(
            call_id=call_id,
            transcript=customer_transcript,
            strategy=strategy,
            customer_only=False,
        )

        return {
            "call_id": call_id,
            "full_transcript": whisper_result["transcript"],
            "customer_speaker": customer_speaker,
            "customer_transcript": customer_transcript,
            "speaker_segments": segments,
            "aligned_words": aligned_words,
            "intent_probabilities": intent_result["intent_probabilities"],
            "intent_predictions": intent_result["intent_predictions"],
        }
