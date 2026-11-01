"""Semantic retrieval for CallSense."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class CallRetrieval:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        required = {"call_id", "whisper_transcript"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        self.df = self.df.dropna(subset=["whisper_transcript"]).copy()
        self.df["transcript"] = (
            self.df["whisper_transcript"].astype(str).str.strip()
        )
        self.df = self.df[self.df["transcript"].ne("")].reset_index(drop=True)

        if self.df.empty:
            raise ValueError("No valid transcripts found")

        self.df["audio_id"] = self.df["call_id"].astype(str)

        self.embedder = SentenceTransformer(
            "intfloat/multilingual-e5-base"
        )

        self.embeddings = self.embedder.encode(
            ["passage: " + text for text in self.df["transcript"]],
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def retrieve(self, query, k=8):
        if not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedder.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:k]

        return [
            {
                "audio_id": self.df.iloc[i]["audio_id"],
                "text": self.df.iloc[i]["transcript"],
                "score": float(scores[i]),
                "dialect": str(self.df.iloc[i].get("dialect", "")),
            }
            for i in indices
        ]
