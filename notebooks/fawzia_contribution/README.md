# CallSense — Intent & Escalation Classification

This directory documents the intent and escalation classification contribution to the CallSense team project.

## Notebooks

| Notebook | Description |
|---|---|
| intent_keyword_logistic_baselines.ipynb | Keyword-based and TF-IDF + Logistic Regression baselines for intent classification |
| intent_camelbert_training.ipynb | CAMeLBERT training for multi-label intent classification |
| escalation_keyword_logistic_baselines.ipynb | Keyword-based and Logistic Regression baselines for escalation detection |
| escalation_camelbert_training.ipynb | CAMeLBERT training for escalation detection |

## Related Components

- `app/streamlit_app.py`: Shared team Streamlit interface integrating multiple CallSense components.
- `scripts/train_camelbert_escalation_full.py`: Escalation model training.
- `scripts/predict_camelbert_escalation_full.py`: Escalation inference.

## Methodology

1. Establish keyword-based baselines.
2. Train TF-IDF and Logistic Regression baselines.
3. Fine-tune CAMeLBERT models.
4. Evaluate classification performance and analyze errors.

## Notes

- Training datasets and model checkpoints are not included in this directory.
- The notebooks may require local project datasets and dependencies.
- Reported development-set results should not be interpreted as independent test-set performance.
- The Streamlit application is a shared team integration, not a standalone implementation of these notebooks.
