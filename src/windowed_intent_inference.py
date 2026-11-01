"""Offline frozen CAMeLBERT inference; no evaluation or dataset access.

128 total tokens, 64 overlapping content tokens (62-token advance for BERT).
Pooling applies to sigmoid probabilities, separately for each intent.
"""
import torch
from src import unified_inference as ui
from src.speaker_turns import extract_customer

LABELS = ui.LABELS
MAX_LENGTH = 128
OVERLAP = 64
STRATEGIES = ('max_pooling', 'mean_pooling')


def build_windows(tokenizer, text):
    """Use tokenizer overflow; retain every content token with specials per window.

    Returns CPU tensors and half-open offsets into the untruncated content IDs.
    Padding is fixed at 128 and excluded by attention masks.
    """
    ui._nonempty(text, 'prepared text')
    if tokenizer.padding_side != 'right' or tokenizer.truncation_side != 'right':
        raise ValueError('Expected right padding and truncation')
    special = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = MAX_LENGTH - special
    advance = capacity - OVERLAP
    if advance <= 0 or tokenizer.pad_token_id is None:
        raise ValueError('Tokenizer cannot support this window configuration')
    encoded = tokenizer(text, max_length=MAX_LENGTH, padding='max_length',
                        truncation=True, stride=OVERLAP,
                        return_overflowing_tokens=True, return_tensors='pt')
    windows, spans = [], []
    for index in range(len(encoded['input_ids'])):
        window = {field: encoded[field][index:index+1].to('cpu')
                  for field in ('input_ids', 'attention_mask')}
        content = int(window['attention_mask'].sum()) - special
        if content <= 0:
            raise ValueError('Input has no content tokens')
        start = index * advance
        windows.append(window)
        spans.append((start, start + content))
    return windows, spans


def pool_probabilities(probabilities, strategy):
    """Max or equal-weight mean across windows, after sigmoid (not logits)."""
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown strategy; choose {STRATEGIES}')
    if (probabilities.ndim != 2 or probabilities.shape[0] == 0
            or probabilities.shape[1] != len(LABELS)
            or not torch.isfinite(probabilities).all().item()
            or ((probabilities < 0) | (probabilities > 1)).any().item()):
        raise ValueError('Expected finite window probabilities with shape [windows, 5]')
    return probabilities.max(dim=0).values if strategy == 'max_pooling' else probabilities.mean(dim=0)


class WindowedIntentPipeline:
    """Load the existing frozen intent checkpoint once, on CPU.

    Saved checkpoint thresholds are preserved. Does not load escalation or MARBERT.
    Customer extraction is opt-in and rejects ambiguous/unsupported input.
    """
    def __init__(self):
        settings = ui._settings(ui.INTENT_CHECKPOINT)
        if (settings.get('intent_cols') != list(LABELS)
                or settings.get('probability_function') != 'sigmoid'
                or settings.get('threshold_comparison') != '>='
                or set(settings.get('thresholds', {})) != set(LABELS)):
            raise ValueError('Unexpected checkpoint settings or labels')
        self.thresholds = {label: ui._threshold(settings['thresholds'][label]) for label in LABELS}
        self.tokenizer, self.model = ui._load(ui.INTENT_CHECKPOINT, dict(enumerate(LABELS)))
        self.checkpoint_sha256 = ui._sha256(ui.INTENT_CHECKPOINT / 'model.safetensors')
        self.settings_sha256 = ui._sha256(ui.INTENT_CHECKPOINT / 'settings.json')

    def predict(self, call_id, transcript, *, strategy, customer_only=False):
        ui._nonempty(call_id, 'call_id')
        ui._nonempty(transcript, 'transcript')
        if strategy not in STRATEGIES:
            raise ValueError(f'Unknown strategy; choose {STRATEGIES}')
        if not isinstance(customer_only, bool):
            raise ValueError('customer_only must be a boolean')
        attribution = None
        if customer_only:
            extracted = extract_customer(transcript)
            if not extracted['valid']:
                raise ValueError('Customer extraction failed: ' + '; '.join(extracted['issues']))
            transcript = extracted['customer_transcript']
            attribution = {k: v for k, v in extracted.items() if k != 'customer_transcript'}
        text = ui.prepare_text(transcript)
        windows, spans = build_windows(self.tokenizer, text)
        probabilities = []
        with torch.inference_mode():
            for window in windows:
                # Same input fields, shape, padding and execution device as baseline.
                logits = self.model(**window).logits
                if tuple(logits.shape) != (1, len(LABELS)) or not torch.isfinite(logits).all().item():
                    raise RuntimeError('Model returned invalid logits')
                probabilities.append(torch.sigmoid(logits)[0])
            pooled = pool_probabilities(torch.stack(probabilities), strategy)
        values = dict(zip(LABELS, pooled.tolist()))
        return {'call_id': call_id, 'intent_probabilities': values,
                'intent_predictions': {l: values[l] >= self.thresholds[l] for l in LABELS},
                'metadata': {'strategy': strategy, 'customer_only': customer_only,
                    'customer_attribution': attribution, 'label_order': list(LABELS),
                    'thresholds': dict(self.thresholds), 'threshold_source': 'saved_checkpoint',
                    'comparison': '>=', 'max_length': MAX_LENGTH, 'overlap_content_tokens': OVERLAP,
                    'advance_content_tokens': MAX_LENGTH-self.tokenizer.num_special_tokens_to_add(False)-OVERLAP,
                    'window_count': len(windows), 'content_token_count': spans[-1][1],
                    'window_content_spans': [list(span) for span in spans],
                    'device': 'cpu', 'preprocessing': ui.PREPROCESSING_VERSION,
                    'checkpoint': str(ui.INTENT_CHECKPOINT), 'checkpoint_sha256': self.checkpoint_sha256,
                    'settings_sha256': self.settings_sha256}}
