"""End-to-end CAMeLBERT escalation experiment. Requires MPS; no threshold search.

The unchanged development set selects the epoch by loss and reports final metrics;
it is therefore development evaluation, never an untouched test evaluation.
"""

import argparse
import gc
import hashlib
import json
import os
import plistlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
BASE_REVISION = "9be352797bdf28a9ae21e2ae582aaaca7abdb22d"
BASE_DIR = (Path.home() / ".cache/huggingface/hub"
            / "models--CAMeL-Lab--bert-base-arabic-camelbert-mix/snapshots" / BASE_REVISION)
BASE_WEIGHTS_SHA256 = "d234e4fe0193e74036b1db72ba9b41a2e4921d8cf0c3c421d1f6a6d8816af729"
TRAIN_PATH = ROOT / "data/train_escalation_adjudicated_20261030T221110Z.csv"
VAL_PATH = ROOT / "data/processed/val.csv"
LR_MODEL = ROOT / "models/escalation_tfidf_adjudicated_20261030T221503Z"
LR_EVAL = ROOT / "eval/escalation_tfidf_adjudicated_20261030T221503Z"
FROZEN_MODEL = ROOT / "models/camelbert_escalation_frozen_adjudicated_20261030T224908_644621Z"
FROZEN_EVAL = ROOT / "eval/camelbert_escalation_frozen_adjudicated_20261030T224908_644621Z"
SEED = 42
MAX_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
MAX_LENGTH = 128
PATIENCE = 2
THRESHOLD = 0.5
WEIGHT_DECAY = 0.01
# Disk budgets only; these do not change model/training settings.
PRETRAIN_OPERATING_RESERVE = 256 * 1024**2
EXPORT_MIN_MARGIN = 128 * 1024**2


def bounded_disk_diagnostic():
    """At most 32 MiB of payload writes; remove only our own temporary files."""
    parent = ROOT / "models"
    before = shutil.disk_usage(parent).free
    if before < 64 * 1024**2:
        raise RuntimeError("Diagnostic requires 64 MiB free; no test files created")
    with tempfile.TemporaryDirectory(prefix=".export_diagnostic_", dir=parent) as name:
        directory = Path(name)
        path = directory / "ordinary.bin"
        chunk = os.urandom(1024**2)
        expected = hashlib.sha256(chunk * 16).hexdigest()
        with path.open("xb") as handle:
            for _ in range(16):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        renamed = directory / "renamed.bin"
        path.rename(renamed)
        if sha256(renamed) != expected:
            raise RuntimeError("Diagnostic write verification failed")
        renamed.unlink()  # Only a file created by this diagnostic.
        tensor = torch.randint(0, 256, (15 * 1024**2,), dtype=torch.uint8,
                               generator=torch.Generator().manual_seed(42))
        path = directory / "synthetic.safetensors"
        save_file({"synthetic": tensor}, path, metadata={"format": "pt"})
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        if path.stat().st_size > 16 * 1024**2:
            raise RuntimeError("Unexpected diagnostic serialization size")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if not torch.equal(handle.get_tensor("synthetic"), tensor):
                raise RuntimeError("Diagnostic safetensors verification failed")
        result = {"ordinary_write_bytes": 16 * 1024**2,
                  "safetensors_write_bytes": path.stat().st_size,
                  "free_before_bytes": before, "status": "passed"}
    result.update(cleaned_up=not directory.exists(),
                  free_after_bytes=shutil.disk_usage(parent).free)
    print(json.dumps(result, indent=2), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_size(state):
    """Estimate one safetensors checkpoint from CPU tensors, without serializing.

    Shared storage is counted once; full storage sizes conservatively cover views.
    Include an allowance for tensor names, shapes and the serialization header.
    """
    storages = {}
    for tensor in state.values():
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
    header_bytes = max(64 * 1024, len(state) * 512)
    return sum(storages.values()) + header_bytes


def system_storage_diagnostics():
    """Read host diagnostics when permitted; never infer availability on failure."""
    result = {}
    for name, command in {
        "apfs": ["diskutil", "apfs", "list", "-plist"],
        "swap": ["sysctl", "vm.swapusage"],
    }.items():
        try:
            process = subprocess.run(command, capture_output=True, timeout=10, check=True)
            if name == "apfs":
                containers = plistlib.loads(process.stdout)["Containers"]
                if not containers or any("CapacityFree" not in item for item in containers):
                    raise ValueError("APFS container free-space fields were not returned")
                result[name] = {"available": True, "containers": [
                    {"container": item.get("ContainerReference"),
                     "capacity_free_bytes": item.get("CapacityFree"),
                     "capacity_ceiling_bytes": item.get("CapacityCeiling")}
                    for item in containers]}
            else:
                result[name] = {"available": True,
                                "reading": process.stdout.decode().strip()}
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
            detail = getattr(error, "stderr", b"") or str(error)
            if isinstance(detail, bytes):
                detail = detail.decode(errors="replace").strip()
            result[name] = {"available": False, "reason": detail}
    return result


def disk_diagnostics(stage, initial_free, checkpoint_bytes=None, evaluation_bytes=0,
                     tokenizer_bytes=0, enforce=False, operating_reserve=0):
    """Check each output filesystem once, combining budgets when it is shared."""
    paths = {"model": ROOT / "models", "evaluation": ROOT / "eval"}
    filesystems = {}
    for kind, path in paths.items():
        fs_id = str(path.stat().st_dev)
        entry = filesystems.setdefault(fs_id, {"paths": [], "payload_bytes": 0})
        entry["paths"].append(str(path))
        if checkpoint_bytes is not None:
            entry["payload_bytes"] += (
                checkpoint_bytes + tokenizer_bytes + 1024**2
                if kind == "model" else evaluation_bytes
            )
    reports = []
    failures = []
    system = system_storage_diagnostics()
    for fs_id, entry in filesystems.items():
        free = shutil.disk_usage(entry["paths"][0]).free
        initial_free.setdefault(fs_id, free)
        margin = (max(EXPORT_MIN_MARGIN, (entry["payload_bytes"] + 4) // 5)
                  if checkpoint_bytes is not None else 0)
        required = entry["payload_bytes"] + margin + operating_reserve
        report = {
            "stage": stage, "filesystem_id": fs_id, "paths": entry["paths"],
            "free_bytes": free, "free_gib": free / 1024**3,
            "change_since_start_bytes": free - initial_free[fs_id],
            "estimated_checkpoint_bytes": checkpoint_bytes,
            "estimated_output_payload_bytes": entry["payload_bytes"],
            "export_safety_margin_bytes": margin,
            "pretraining_operating_reserve_bytes": operating_reserve,
            "required_bytes": required,
            "sufficient": checkpoint_bytes is None or free >= required,
            "system_storage": system,
        }
        reports.append(report)
        print("DISK " + json.dumps(report), flush=True)
        if not report["sufficient"]:
            failures.append(f"{entry['paths']}: {free / 1024**3:.3f} GiB free, "
                            f"{required / 1024**3:.3f} GiB required; "
                            f"short by {(required - free) / 1024**2:.1f} MiB")
    if failures:
        message = "Disk budget insufficient: " + "; ".join(failures)
        if enforce:
            raise RuntimeError(message + ". No files deleted or alternate destination selected.")
        print("WARNING: " + message, flush=True)
    return reports


def write_json_new(path, value):
    """Exclusive creation: neither recovery nor status updates overwrite a file."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def export_event(eval_dir, model_dir, attempt, status, copy_to_model=True, **details):
    event = {"status": status, "attempt": attempt, "model_directory": str(model_dir),
             "utc": datetime.now(timezone.utc).isoformat(), **details}
    # Append-only event files; a verified marker is created only after all checks.
    name = f"export_status_{attempt:02d}_{status}.json"
    write_json_new(eval_dir / name, event)
    if copy_to_model:
        write_json_new(model_dir / name, event)
    print("EXPORT " + json.dumps(event), flush=True)
    return event


def verify_checkpoint(directory, selected_state):
    """Reload HF config/tokenizer and assign mmap tensors to a meta model.

    Meta initialization allocates no second initialized parameter copy. Tensor
    equality and strict assignment verify every exported parameter/buffer.
    A tiny synthetic forward checks that the reconstructed model can execute.
    """
    path = directory / "model.safetensors"
    digest = sha256(path)
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        reloaded = AutoModelForSequenceClassification.from_config(config)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(selected_state):
            raise RuntimeError("Checkpoint tensor names do not match selected weights")
        mapped = {}
        for name, expected in selected_state.items():
            tensor = handle.get_tensor(name)
            if tensor.dtype != expected.dtype or tensor.shape != expected.shape:
                raise RuntimeError("Checkpoint dtype/shape mismatch: " + name)
            if not torch.equal(tensor, expected):
                raise RuntimeError("Checkpoint tensor differs from selected weights: " + name)
            mapped[name] = tensor
        reloaded.load_state_dict(mapped, strict=True, assign=True)
        del mapped
        # BERT regenerates these non-persistent buffers instead of saving them.
        for module in reloaded.modules():
            for name in module._non_persistent_buffers_set:
                buffer = module._buffers.get(name)
                if buffer is not None and buffer.is_meta:
                    if name == "position_ids":
                        module._buffers[name] = torch.arange(buffer.numel(), dtype=buffer.dtype).reshape(buffer.shape)
                    elif name == "token_type_ids":
                        module._buffers[name] = torch.zeros(buffer.shape, dtype=buffer.dtype)
        if any(t.is_meta for t in list(reloaded.parameters()) + list(reloaded.buffers())):
            raise RuntimeError("Reload left uninitialized meta tensors")
        reloaded.eval()
        encoded = tokenizer("اختبار", return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits = reloaded(**encoded).logits
        if logits.shape != (1, 2) or not torch.isfinite(logits).all().item():
            raise RuntimeError("Reloaded model synthetic forward failed")
        del reloaded, logits, encoded, tensor
    gc.collect()
    return {"checkpoint_sha256": digest, "checkpoint_bytes": path.stat().st_size,
            "verification": "all tensors equal; strict HF meta-model reload; finite synthetic CPU forward"}


def normalize_cpu_state_in_place(state):
    """Replace only strided CPU tensors, preserving values and dictionary identity.

    A CPU copy can preserve a tensor's non-contiguous layout. Safetensors needs
    contiguous storage. Replace one entry at a time rather than copying the
    entire checkpoint; already contiguous tensors keep their existing storage.
    """
    changed = 0
    for name in state:
        tensor = state[name]
        if tensor.device.type != "cpu":
            raise ValueError("Selected checkpoint must contain CPU tensors: " + name)
        if not tensor.is_contiguous():
            state[name] = tensor.contiguous()
            changed += 1
        del tensor
    return changed


def export_with_recovery(state, config, tokenizer, settings, model_dir, eval_dir,
                         prompt=input):
    """At most three export attempts, each in a new directory; never retrain.

    Interactive input deliberately keeps CPU weights alive while the operator
    chooses retry, an alternate destination parent, or stop. After the third
    failure, acknowledgement is required before releasing the weights.
    """
    original_dir = model_dir
    for attempt in range(1, 4):
        created = False
        try:
            model_dir.mkdir(exist_ok=False)
            created = True
            export_event(eval_dir, model_dir, attempt, "pending")
            config.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            write_json_new(model_dir / "settings.json", settings)
            free = shutil.disk_usage(model_dir).free
            needed = checkpoint_size(state)
            required = needed + max(EXPORT_MIN_MARGIN, (needed + 4) // 5)
            diagnostic = {"free_bytes": free, "required_bytes": required,
                          "system_storage": system_storage_diagnostics()}
            write_json_new(eval_dir / f"export_disk_{attempt:02d}.json", diagnostic)
            print("EXPORT_DISK " + json.dumps(diagnostic), flush=True)
            if free < required:
                raise RuntimeError(f"Export budget insufficient: {free} free, {required} required")
            # No model.to('cpu'), initialized model copy, or optimizer checkpoint.
            normalized = normalize_cpu_state_in_place(state)
            print(f"Export layout check: normalized {normalized} CPU tensors in place", flush=True)
            save_file(state, model_dir / "model.safetensors", metadata={"format": "pt"})
            with (model_dir / "model.safetensors").open("rb") as handle:
                os.fsync(handle.fileno())
            verification = verify_checkpoint(model_dir, state)
            export_event(eval_dir, model_dir, attempt, "verified", **verification)
            return model_dir
        except Exception as error:
            # Only text escapes this block: retaining a traceback could retain
            # the verification model and its mapped checkpoint tensors.
            failure = f"{type(error).__name__}: {error}"
        gc.collect()
        print("Export failed; selected CPU weights are still in memory: " + failure, flush=True)
        try:
            export_event(eval_dir, model_dir, attempt, "failed", copy_to_model=created, error=failure)
        except OSError as status_error:
            print(f"Could not persist failed status: {status_error}. No verified marker exists.", flush=True)
        if attempt == 3:
            prompt("Three export attempts exhausted. CPU weights remain alive until you press Enter to stop: ")
            raise RuntimeError("Export not verified after three attempts: " + failure)
        parent = None
        for _ in range(3):
            answer = prompt(
                "Weights retained. Enter retry, an EXISTING alternate parent directory "
                "(prefer a different disk), or stop. No files will be deleted: ").strip()
            if answer.lower() == "stop":
                raise RuntimeError("Operator stopped export recovery: " + failure)
            candidate_parent = original_dir.parent if answer.lower() == "retry" else Path(answer).expanduser()
            if answer and candidate_parent.is_dir():
                parent = candidate_parent
                break
            print("Invalid destination; weights remain alive. Use retry, stop, or an existing directory.", flush=True)
        if parent is None:
            prompt("Three invalid recovery responses. Press Enter to stop and release CPU weights: ")
            raise RuntimeError("No valid recovery destination supplied")
        # Failed attempts are preserved. Even retries use fresh directories.
        model_dir = parent / ("camelbert_escalation_full_" +
                              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    raise AssertionError("Unreachable")


def normalize_for_check(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", text)
    text = re.sub("[\u064b-\u065f\u0670\u0640]", "", text)
    text = re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def validate(frame, name, count, positives):
    if not {"complaint_id", "model_text", "escalation"} <= set(frame.columns):
        raise ValueError(f"{name}: missing columns")
    if len(frame) != count or frame.complaint_id.isna().any() or frame.complaint_id.duplicated().any():
        raise ValueError(f"{name}: incorrect row count or invalid IDs")
    if not frame.complaint_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name}: empty/invalid IDs")
    if not frame.model_text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{name}: missing text")
    if not frame.escalation.isin([0, 1]).all() or int(frame.escalation.sum()) != positives:
        raise ValueError(f"{name}: unexpected label distribution")


def metrics(actual, probabilities):
    predicted = (probabilities >= THRESHOLD).astype(int)
    return predicted, {
        "accuracy": float(accuracy_score(actual, predicted)),
        "positive_precision": float(precision_score(actual, predicted, zero_division=0)),
        "positive_recall": float(recall_score(actual, predicted, zero_division=0)),
        "positive_f1": float(f1_score(actual, predicted, zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(actual, probabilities)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(actual, predicted, labels=[0, 1],
            target_names=["no_likely_escalation", "likely_escalation"], output_dict=True, zero_division=0),
    }


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("Run in an interactive terminal so failed exports can retain weights for recovery")
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json",
                      "special_tokens_map.json", "vocab.txt"]
    missing = [str(BASE_DIR / name) for name in required_files if not (BASE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Original base checkpoint files unavailable: " + ", ".join(missing))
    config = read_json(BASE_DIR / "config.json")
    if config.get("architectures") != ["BertForMaskedLM"] or config.get("hidden_size") != 768:
        raise ValueError("Expected original pretrained MLM config, not a fine-tuned classifier")
    if sha256(BASE_DIR / "pytorch_model.bin") != BASE_WEIGHTS_SHA256:
        raise ValueError("Original pretrained base weights differ from inspected local fingerprint")
    initial_free = {}
    disk_history = disk_diagnostics("startup", initial_free)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required; execute in the working local CallSense environment")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable CPU fallback and restart; MPS is required")
    device = torch.device("mps")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH)
    validate(train, "training", 124, 41)
    validate(validation, "validation", 31, 5)
    if set(train.complaint_id) & set(validation.complaint_id):
        raise ValueError("Cross-split ID overlap")
    for fn in [lambda text: text, normalize_for_check]:
        if set(train.model_text.map(fn)) & set(validation.model_text.map(fn)):
            raise ValueError("Exact/normalized cross-split text overlap")
    input_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [TRAIN_PATH, VAL_PATH]}
    for model_dir, eval_dir in [(LR_MODEL, LR_EVAL), (FROZEN_MODEL, FROZEN_EVAL)]:
        previous = read_json(model_dir / "settings.json")
        for name, digest in input_hashes.items():
            if previous["input_sha256"].get(name) != digest:
                raise ValueError("Comparison dataset hash mismatch: " + str(model_dir))
        prior_predictions = pd.read_csv(eval_dir / "predictions.csv")
        for column in ["complaint_id", "model_text", "escalation"]:
            if prior_predictions[column].tolist() != validation[column].tolist():
                raise ValueError("Comparison validation alignment mismatch: " + str(eval_dir))
    protected = [TRAIN_PATH, VAL_PATH, *[BASE_DIR / name for name in required_files]]
    protected += [d / name for d in [LR_MODEL, FROZEN_MODEL] for name in ["settings.json"]]
    protected += [d / name for d in [LR_EVAL, FROZEN_EVAL] for name in ["metrics.json", "predictions.csv"]]
    source_hashes = {str(p): sha256(p) for p in protected}

    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_DIR, local_files_only=True, num_labels=2,
        id2label={0: "no_likely_escalation", 1: "likely_escalation"},
        label2id={"no_likely_escalation": 0, "likely_escalation": 1},
        problem_type="single_label_classification",
    )
    if not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("End-to-end model must have all parameters trainable")
    # Estimate actual classifier tensors on CPU, not the larger pretrained MLM file.
    estimated_checkpoint = checkpoint_size(model.state_dict())
    tokenizer_allowance = max(8 * 1024**2, 2 * sum(
        (BASE_DIR / name).stat().st_size for name in required_files if name != "pytorch_model.bin"
    ))
    evaluation_allowance = max(8 * 1024**2, 3 * VAL_PATH.stat().st_size + 1024**2)
    disk_history += disk_diagnostics(
        "before_training", initial_free, estimated_checkpoint,
        evaluation_allowance, tokenizer_allowance, enforce=True,
        operating_reserve=PRETRAIN_OPERATING_RESERVE,
    )
    model.to(device)

    def dataset(frame):
        encoded = tokenizer(frame.model_text.tolist(), padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        return TensorDataset(encoded["input_ids"], encoded["attention_mask"],
                             torch.tensor(frame.escalation.to_numpy(), dtype=torch.long))

    train_loader = DataLoader(dataset(train), batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(dataset(validation), batch_size=BATCH_SIZE, shuffle=False)
    counts = np.bincount(train.escalation.to_numpy(dtype=int), minlength=2)
    class_weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history, best_state = [], None
    best_loss, best_epoch, wait = float("inf"), None, 0

    def pass_loss(loader, training):
        model.train(training)
        total, denominator = 0.0, 0.0
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = criterion(logits, labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss; stopping without configuration changes")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
            # Weighted CE mean divides by sum of target weights, not batch size.
            batch_weight = class_weights[labels].sum().item()
            total += loss.item() * batch_weight
            denominator += batch_weight
        return total / denominator

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = pass_loss(train_loader, True)
        val_loss = pass_loss(val_loader, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"Epoch {epoch}/{MAX_EPOCHS}: train={train_loss:.6f}, validation={val_loss:.6f}", flush=True)
        disk_history += disk_diagnostics(
            f"after_epoch_{epoch}", initial_free, estimated_checkpoint,
            evaluation_allowance, tokenizer_allowance,
        )
        if val_loss < best_loss:
            candidate = {name: value.detach().to(device="cpu", copy=True,
                                                memory_format=torch.contiguous_format)
                         for name, value in model.state_dict().items()}
            if not all(torch.isfinite(v).all().item() for v in candidate.values() if v.is_floating_point()):
                raise RuntimeError("Invalid checkpoint weights")
            best_state, best_loss, best_epoch, wait = candidate, val_loss, epoch, 0
            del candidate  # best_state is the only retained selected snapshot.
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No valid best checkpoint")
    model.load_state_dict(best_state)
    selected_checkpoint_bytes = checkpoint_size(best_state)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()
    probabilities = []
    with torch.no_grad():
        for ids, mask, _ in val_loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            scores = torch.softmax(logits, dim=-1)[:, 1]
            if not torch.isfinite(scores).all().item():
                raise RuntimeError("Non-finite evaluation probabilities")
            probabilities.append(scores.cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predicted, report = metrics(validation.escalation.to_numpy(dtype=int), probabilities)
    comparison = {
        "camelbert_full": report,
        "adjudicated_logistic_regression": read_json(LR_EVAL / "metrics.json")["new_adjudicated"],
        "adjudicated_frozen_head": read_json(FROZEN_EVAL / "metrics.json")["frozen_camelbert_adjudicated"],
        "thresholds": {"camelbert_full": THRESHOLD,
                       "logistic_regression": read_json(LR_MODEL / "settings.json")["threshold"],
                       "frozen_head": read_json(FROZEN_MODEL / "settings.json")["threshold"]},
        "comparison_note": "Same data/labels, different fixed thresholds. Full model selects its epoch on development loss; prior heads used training-only selection. Not a controlled architecture-only or untouched-test comparison.",
    }
    for name, digest in source_hashes.items():
        if sha256(Path(name)) != digest:
            raise RuntimeError("Read-only source changed: " + name)
    # Retain only the selected CPU snapshot. Release model/gradients/device tensors
    # before disk export; do not create another CPU model via model.to('cpu').
    model_config = model.config
    model_config.architectures = [type(model).__name__]
    saved_class_weights = class_weights.cpu().tolist()
    torch.mps.synchronize()
    del logits, scores, ids, mask, model, criterion, class_weights
    del pass_loss, train_loader, val_loader
    gc.collect()
    torch.mps.empty_cache()
    disk_history += disk_diagnostics(
        "after_device_cleanup_before_export", initial_free, selected_checkpoint_bytes,
        evaluation_allowance, tokenizer_allowance, enforce=False,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    model_dir = ROOT / "models" / f"camelbert_escalation_full_{run_id}"
    eval_dir = ROOT / "eval" / f"camelbert_escalation_full_{run_id}"
    if model_dir.exists() or eval_dir.exists():
        raise FileExistsError("Timestamped output collision")
    settings = {
        "model_name": MODEL_NAME, "base_revision": BASE_REVISION, "base_directory": str(BASE_DIR),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "target": "escalation", "num_labels": 2, "trainable_encoder": True,
        "seed": SEED, "device": "mps", "epochs_max": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
        "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": 1.0,
        "loss": "CrossEntropyLoss", "class_weight_rule": "N / (2 * class_count), training only",
        "class_weights": saved_class_weights, "training_class_counts": counts.tolist(),
        "training_records": len(train), "validation_records": len(validation),
        "threshold": THRESHOLD, "comparison": ">=", "probability_function": "softmax(logits)[:, 1]",
        "early_stopping": {"criterion": "minimum weighted development loss", "patience": PATIENCE},
        "best_epoch": best_epoch, "best_validation_loss": best_loss, "history": history,
        "text_preprocessing": "Unmodified model_text, saved tokenizer, max_length padding and truncation",
        "export_protocol": {"initial_status": "pending", "evaluation_directory": str(eval_dir),
                            "status_files": "export_status_<attempt>_<pending|failed|verified>.json",
                            "success_rule": "Use only an attempt with a verified event; pending is incomplete",
                            "maximum_attempts": 3, "checkpoint_format": "safetensors, metadata format=pt"},
        "training_texts_over_max_length": sum(len(tokenizer.encode(t, truncation=False)) > MAX_LENGTH for t in train.model_text),
        "disk_space": {
            "pretraining_operating_reserve_bytes": PRETRAIN_OPERATING_RESERVE,
            "selected_checkpoint_estimate_bytes": selected_checkpoint_bytes,
            "tokenizer_allowance_bytes": tokenizer_allowance,
            "evaluation_allowance_bytes": evaluation_allowance,
            "diagnostics": disk_history,
            "note": "Disk availability can change after checks; reserves cannot guarantee future space or bound macOS swap growth.",
        },
        "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "sklearn": sklearn.__version__},
        "limitations": ["Development selects epoch and has been inspected previously; not an untouched test set.",
                        "Validation retains original labels; training has adjudicated labels.",
                        "Known source/author provenance issues remain; metadata is not used as features.",
                        "128-token truncation preserves initial settings but may lose late evidence."],
    }
    eval_dir.mkdir(exist_ok=False)
    # Small results survive checkpoint failure. Status events are immutable and
    # always identify the attempted model path, including an alternate volume.
    write_json_new(eval_dir / "settings.json", settings)
    write_json_new(eval_dir / "metrics.json", comparison)
    results = validation[["complaint_id", "model_text", "escalation"]].copy()
    results["escalation_probability"] = probabilities
    results["escalation_prediction"] = predicted
    results["error_type"] = np.where((validation.escalation.to_numpy() == 0) & (predicted == 1), "false_positive",
                                     np.where((validation.escalation.to_numpy() == 1) & (predicted == 0), "false_negative", ""))
    results.to_csv(eval_dir / "predictions.csv", index=False, mode="x")
    results.loc[results.error_type.ne("")].to_csv(eval_dir / "errors.csv", index=False, mode="x")
    pd.DataFrame(history).to_csv(eval_dir / "training_history.csv", index=False, mode="x")
    for path in eval_dir.iterdir():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    print("Evaluation saved; model export pending:", eval_dir, flush=True)
    model_dir = export_with_recovery(best_state, model_config, tokenizer, settings,
                                     model_dir, eval_dir)
    del best_state
    gc.collect()
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print("Best epoch:", best_epoch, "Validation loss:", best_loss)
    print("Model:", model_dir)
    print("Evaluation:", eval_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-diagnostic", action="store_true",
                        help="Run only the bounded, self-cleaning 32 MiB disk diagnostic")
    arguments = parser.parse_args()
    if arguments.disk_diagnostic:
        bounded_disk_diagnostic()
    else:
        main()
