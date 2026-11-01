"""Reusable offline CPU inference; no dataset, training, or evaluation imports."""

import copy
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path

# Set before Hugging Face imports; every loader also requires local files.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INTENT_CHECKPOINT = ROOT / "models/camelbert_intent_reviewed_20261030T150108_941481Z"
ESCALATION_CHECKPOINT = ROOT / "models/camelbert_escalation_full_20261030T235430_667676Z"
LABELS = ("billing", "technical", "cancellation", "complaint", "information")
ESCALATION_LABELS = {0: "no_likely_escalation", 1: "likely_escalation"}
MAX_LENGTH = 128
PREPROCESSING_VERSION = "camelbert_model_text_v1"


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def prepare_text(transcript: str) -> str:
    """Match notebook 01 cells 15 and 30: cleaned_text then reference masking.

    Retains ta marbuta, case, punctuation, and speaker content. Does not mutate
    the original string or perform TF-IDF normalization.
    """
    _nonempty(transcript, "transcript")
    text = re.sub(r"https?://\S+|www\.\S+", " ", transcript)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b[A-Za-z]{2,6}\d{6,}\b", "[COMPLAINT_REF]", text,
                  flags=re.IGNORECASE)
    _nonempty(text, "prepared transcript")
    return text


def _threshold(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Threshold must be a finite number between 0 and 1")
    return float(value)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings(directory):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors", "settings.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    settings = json.loads((directory / "settings.json").read_text(encoding="utf-8"))
    if settings.get("max_length") != MAX_LENGTH:
        raise ValueError("Expected saved max_length=128")
    return settings


def _load(directory, labels):
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    with torch.device("cpu"):
        model, info = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, output_loading_info=True)
    if any(info.get(key) for key in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete or incompatible checkpoint: {directory}")
    if (model.config.num_labels != len(labels) or model.config.id2label != labels
            or model.config.label2id != {label: i for i, label in labels.items()}):
        raise ValueError(f"Unexpected output label mapping: {directory}")
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


class UnifiedInferencePipeline:
    """Load the fixed checkpoint pair once; reuse this instance across calls."""

    def __init__(self, *, escalation_threshold=None):
        override = None if escalation_threshold is None else _threshold(escalation_threshold)
        intent = _settings(INTENT_CHECKPOINT)
        escalation = _settings(ESCALATION_CHECKPOINT)
        thresholds = intent.get("thresholds", {})
        if (intent.get("intent_cols") != list(LABELS)
                or set(thresholds) != set(LABELS)
                or intent.get("probability_function") != "sigmoid"
                or intent.get("threshold_comparison") != ">="):
            raise ValueError("Invalid saved Intent labels, thresholds, or decision rule")
        self.intent_thresholds = {label: _threshold(thresholds[label]) for label in LABELS}
        if (escalation.get("threshold") != 0.50
                or escalation.get("comparison") != ">="
                or escalation.get("probability_function") != "softmax(logits)[:, 1]"):
            raise ValueError("Unexpected saved Escalation decision rule")
        self.escalation_threshold = 0.50 if override is None else override
        self.threshold_source = "saved_checkpoint" if override is None else "explicit_override"
        hashes = {"intent": _sha256(INTENT_CHECKPOINT / "model.safetensors"),
                  "escalation": _sha256(ESCALATION_CHECKPOINT / "model.safetensors")}
        marker = json.loads((ESCALATION_CHECKPOINT / "export_status_01_verified.json").read_text())
        if marker.get("status") != "verified" or marker.get("checkpoint_sha256") != hashes["escalation"]:
            raise ValueError("Escalation checkpoint verification marker/hash mismatch")
        self.intent_tokenizer, self.intent_model = _load(INTENT_CHECKPOINT, dict(enumerate(LABELS)))
        self.escalation_tokenizer, self.escalation_model = _load(ESCALATION_CHECKPOINT, ESCALATION_LABELS)
        self._metadata = {
            "checkpoints": {
                name: {"directory": str(path), "sha256": hashes[name],
                       "settings_sha256": _sha256(path / "settings.json")}
                for name, path in (("intent", INTENT_CHECKPOINT), ("escalation", ESCALATION_CHECKPOINT))
            },
            "intent_label_order": list(LABELS),
            "intent_thresholds": dict(self.intent_thresholds),
            "intent_threshold_source": "saved_checkpoint",
            "escalation_threshold": self.escalation_threshold,
            "escalation_saved_threshold": 0.50,
            "escalation_threshold_source": self.threshold_source,
            "escalation_positive_class_index": 1,
            "threshold_comparison": ">=",
            "probability_functions": {"intent": "sigmoid", "escalation": "softmax(logits)[:, 1]"},
            "preprocessing": {"version": PREPROCESSING_VERSION,
                              "source": "notebooks/01_data_exploration.ipynb cells 15 and 30"},
            "runtime": {"device": "cpu", "python": platform.python_version(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "offline": True},
            "policy_status": "provisional_pending_final_evaluation",
        }

    @staticmethod
    def _infer(text, tokenizer, model, num_labels):
        # Length is for prepared model input, before truncation, including CLS/SEP.
        length = len(tokenizer.encode(text, add_special_tokens=True, truncation=False,
                                      padding=False))
        encoded = tokenizer(text, padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        logits = model(input_ids=encoded["input_ids"].to("cpu"),
                       attention_mask=encoded["attention_mask"].to("cpu")).logits
        if tuple(logits.shape) != (1, num_labels) or not torch.isfinite(logits).all().item():
            raise RuntimeError("Model returned invalid logits")
        return logits, {"original_token_length": length, "max_length": MAX_LENGTH,
                        "retained_token_length": min(length, MAX_LENGTH),
                        "truncated": length > MAX_LENGTH,
                        "removed_token_count": max(0, length - MAX_LENGTH),
                        "includes_special_tokens": True, "padding": "max_length",
                        "padding_side": "right", "truncation_side": "right"}

    def predict(self, call_id: str, transcript: str) -> dict:
        _nonempty(call_id, "call_id")
        text = prepare_text(transcript)
        with torch.inference_mode():
            intent_logits, intent_tokens = self._infer(text, self.intent_tokenizer, self.intent_model, 5)
            escalation_logits, escalation_tokens = self._infer(
                text, self.escalation_tokenizer, self.escalation_model, 2)
            probabilities = dict(zip(LABELS, torch.sigmoid(intent_logits)[0].tolist()))
            escalation_probability = torch.softmax(escalation_logits, dim=-1)[0, 1].item()
        metadata = copy.deepcopy(self._metadata)
        metadata.update({"original_transcript": transcript, "prepared_text": text,
                         "tokenization": {"intent": intent_tokens, "escalation": escalation_tokens}})
        return {"call_id": call_id,
                "intent_predictions": {label: probabilities[label] >= self.intent_thresholds[label]
                                       for label in LABELS},
                "intent_probabilities": probabilities,
                "escalation_probability": escalation_probability,
                "escalation_flag": escalation_probability >= self.escalation_threshold,
                "metadata": metadata}
