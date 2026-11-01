"""CPU-only, offline escalation inference. Prints JSON; never writes artifacts.

Pass one or more quoted complaints as positional arguments. Proposed default
0.20 improved validation positive F1 (0.5882 vs 0.5455) and recall (1.00 vs
0.60), accepting more false positives to reduce missed escalations. Selection
is exploratory on a small validation set, not a final test result. Threshold
0.50 remains available as the original comparison. See
docs/camelbert_full_inference_threshold.md for the decision rationale.
"""

import argparse
import json
import os
from pathlib import Path

# Enforce offline mode before importing Hugging Face libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/camelbert_escalation_full_20261030T235430_667676Z"
PROPOSED_THRESHOLD = 0.20
ORIGINAL_THRESHOLD = 0.50


def predict(complaints, model_dir=DEFAULT_MODEL, threshold=PROPOSED_THRESHOLD):
    """Load once and infer one complaint at a time to limit peak CPU memory."""
    if not complaints or any(not isinstance(text, str) or not text.strip() for text in complaints):
        raise ValueError("Provide at least one non-empty complaint")
    if threshold not in (PROPOSED_THRESHOLD, ORIGINAL_THRESHOLD):
        raise ValueError("Choose 0.20 (proposed exploratory default) or 0.50 (original)")
    model_dir = Path(model_dir).expanduser().resolve()
    for name in ["config.json", "tokenizer.json", "model.safetensors", "settings.json"]:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Required local file missing: {model_dir / name}")
    settings = json.loads((model_dir / "settings.json").read_text(encoding="utf-8"))
    max_length = settings["max_length"]
    if max_length != 128 or settings["threshold"] != 0.50:
        raise ValueError("Expected the saved 128-token, original-threshold-0.50 experiment")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False)
    with torch.device("cpu"):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True)
    model.to("cpu")
    model.eval()
    model.requires_grad_(False)
    if (model.config.num_labels != 2
            or model.config.id2label != {0: "no_likely_escalation", 1: "likely_escalation"}):
        raise ValueError("Unexpected binary model label order")

    results = []
    with torch.inference_mode():
        for text in complaints:
            # Match training: unchanged text and max_length padding/truncation.
            batch = tokenizer(text, padding="max_length", truncation=True,
                              max_length=max_length, return_tensors="pt")
            batch = {name: tensor.to("cpu") for name, tensor in batch.items()}
            # Training supplied only these two tokenizer fields to the model.
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            if logits.shape != (1, 2) or not torch.isfinite(logits).all().item():
                raise RuntimeError("Model returned invalid binary logits")
            probability = torch.softmax(logits, dim=-1)[0, 1].item()
            label = lambda threshold: ("Likely Escalation" if probability >= threshold
                                       else "No Likely Escalation")
            results.append({
                "text": text,
                "positive_class_escalation_probability": probability,
                "prediction": label(threshold),
                "prediction_0.50_original": label(ORIGINAL_THRESHOLD),
                "prediction_0.20_exploratory": label(PROPOSED_THRESHOLD),
            })
    return {
        "model_directory": str(model_dir), "device": "cpu", "max_length": max_length,
        "threshold": threshold,
        "proposed_default_threshold": PROPOSED_THRESHOLD,
        "original_comparison_threshold": ORIGINAL_THRESHOLD,
        "threshold_note": "0.20 proposed exploratory default; 0.50 original comparison; not a final test result",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("complaints", nargs="+", help="One or more quoted Arabic complaints")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL,
                        help="Existing local model directory; downloads are disabled")
    parser.add_argument("--threshold", type=float, choices=[PROPOSED_THRESHOLD, ORIGINAL_THRESHOLD],
                        default=PROPOSED_THRESHOLD,
                        help="Default: 0.20 (proposed exploratory); use 0.50 for the original comparison")
    args = parser.parse_args()
    result = predict(args.complaints, args.model_dir, args.threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
