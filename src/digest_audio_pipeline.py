"""Retrieval -> Grounded Digest -> Arabic TTS."""

from src.retrieval_inference import CallRetrieval
from src.grounded_digest_inference import GroundedDigest
from src.digest_tts_pipeline import DigestTTSPipeline


class DigestAudioPipeline:
    def __init__(self, csv_path):
        self.retrieval = CallRetrieval(csv_path)
        self.digest = GroundedDigest()
        self.tts = DigestTTSPipeline()

    def predict(self, query, k=8, output_path="digest.wav"):
        sources = self.retrieval.retrieve(query, k=k)

        digest_result = self.digest.generate(
            query=query,
            sources=sources,
        )

        citation_check = digest_result["citation_check"]

        if (
            not citation_check["has_at_least_one_citation"]
            or citation_check["invalid_citations"]
        ):
            raise ValueError(
                "Digest citations need review before audio generation"
            )

        audio_result = self.tts.predict(
            digest_text=digest_result["digest_text"],
            output_path=output_path,
        )

        return {
            **digest_result,
            "audio_path": audio_result["audio_path"],
        }
