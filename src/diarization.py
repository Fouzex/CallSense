# فصل المتحدثين باستخدام pyannote والبصمات الصوتية وKMeans

import subprocess
import numpy as np
import torch
from sklearn.cluster import KMeans
from pyannote.audio import Pipeline


class SpeakerDiarizer:
    def __init__(self, hf_token):
        # تحميل المودل مرة واحدة لإعادة استخدامه مع المكالمات
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token
        )
        self.embedding_model = self.pipeline._embedding

    def process(self, audio_path, num_speakers=2):
        # تحويل التسجيل إلى صوت أحادي بتردد 16000 باستخدام FFmpeg
        audio_bytes = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(audio_path),
            "-f", "f32le", "-ac", "1", "-ar", "16000", "-"
        ])
        audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()

        # استخراج المقاطع الزمنية للكلام باستخدام pyannote
        result = self.pipeline(
            {
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": 16000
            },
            num_speakers=num_speakers
        )

        # استخراج بصمة صوتية لكل مقطع صالح
        embeddings = []
        valid_segments = []

        for turn, _ in result.speaker_diarization:
            clip = audio[int(turn.start * 16000):int(turn.end * 16000)]

            if len(clip) < 16000:
                continue

            waveform = torch.from_numpy(
                clip.copy()
            ).float().unsqueeze(0).unsqueeze(0)

            embedding = self.embedding_model(waveform)

            embeddings.append(np.asarray(embedding).flatten())
            valid_segments.append(turn)

        # إذا لم توجد مقاطع كافية، نرجع نتائج pyannote الأصلية
        if len(valid_segments) < num_speakers:
            return [
                {
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker
                }
                for turn, speaker in result.speaker_diarization
            ]

        # تجميع البصمات الصوتية حسب المتحدث
        labels = KMeans(
            n_clusters=num_speakers,
            random_state=42,
            n_init=10
        ).fit_predict(np.array(embeddings))

        # تجهيز النتائج لاستخدامها مع Whisper وبقية النظام
        return [
            {
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": f"SPEAKER_{label:02d}"
            }
            for turn, label in zip(valid_segments, labels)
        ]