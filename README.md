# CallSense

**Arabic Customer-Service Call Intelligence | AIEP — Track E**

CallSense is a team-built prototype for analyzing Arabic telecom customer-service conversations. It combines speech recognition, customer-intent classification, escalation detection, semantic retrieval, evidence-grounded summarization, and Arabic text-to-speech in a Streamlit interface.

> **Project status:** Research and demonstration prototype. The team demonstrated the integrated Streamlit workflow; the models and individual components have different levels of evaluation and are not presented as production-ready.

## What CallSense Does

1. **Process a recorded call:** Detect speech regions and transcribe Arabic audio with Whisper.
2. **Understand the conversation:** Predict customer intents and whether the case may require escalation.
3. **Find related cases:** Retrieve semantically similar transcripts using multilingual embeddings and FAISS.
4. **Generate a grounded digest:** Summarize relevant call information with source references.
5. **Play an Arabic summary:** Convert generated text into speech.

```text
Recorded Arabic call
       |
       v
Speech detection (Silero VAD) + ASR (Whisper Small)
       |
       v
Transcript / timestamps
       |------------------------------|
       v                              v
Intent + escalation             Embeddings + FAISS
classification                  similar-call retrieval
       |                              |
       |---------------|--------------|
                       v
            Qwen 2.5 grounded digest
                       |
                       v
                 Arabic MMS-TTS
                       |
                       v
                Streamlit interface
```

## Core Components

| Component | Technology / approach | Purpose |
|---|---|---|
| Arabic speech recognition | OpenAI Whisper (Tiny, Base, Small evaluated) | Transcribe calls |
| Speech activity detection | Silero VAD | Identify speech intervals |
| Speaker attribution | Exploratory speaker labeling; pyannote investigated | Distinguish speakers where feasible |
| Intent classification | Keyword rules, TF-IDF + Logistic Regression, CAMeLBERT | Identify customer needs |
| Escalation detection | Logistic Regression and CAMeLBERT-based approaches | Flag potentially unresolved or urgent cases |
| Privacy exploration | Rules and model-based PII masking | Identify potential personal information |
| Semantic retrieval | `intfloat/multilingual-e5-base` + FAISS | Find relevant prior call transcripts |
| Grounded summarization | Qwen 2.5 | Produce an Arabic digest with source markers |
| Arabic speech synthesis | `facebook/mms-tts-ara` | Read the digest aloud |
| User interface | Streamlit | Bring the pipeline together |

### Intent Categories

The multi-label intent classifier covers **billing**, **technical support**, **cancellation**, **complaints**, and **information/inquiries**. A single conversation can receive more than one label.

## Data

| Source | Prepared collection | Used in initial Whisper evaluation |
|---|---:|---:|
| Common Voice Arabic | 50 recordings | 50 |
| MGB-2 Arabic | 100 recordings | 100 |
| Team-recorded simulated calls | 60 recordings | 49 |
| **Total evaluated** | | **199** |

The team calls are **scripted simulations**, not real customer recordings. They include Modern Standard Arabic, Saudi Arabic, and Arabic–English code-switching. Reference transcripts were checked against the audio. A separate Arabic telecom-complaint text dataset supported the intent and escalation classification experiments.

**Data availability:** Raw audio, private transcripts, model checkpoints, and other sensitive or large artifacts may not be committed to this repository. Access to public datasets remains subject to their respective licenses and terms.

## Evaluation Results

### Whisper Word Error Rate (WER)

Lower WER indicates fewer transcription errors; **WER is not accuracy**.

| Dataset | Whisper Tiny | Whisper Base | Whisper Small |
|---|---:|---:|---:|
| Common Voice (50) | 91.63% | 85.71% | **55.17%** |
| MGB-2 (100) | 70.75% | 53.36% | **34.39%** |
| Team calls (49) | 90.14% | 80.27% | **65.93%** |
| **Overall (199)** | **86.03%** | **74.67%** | **58.77%** |

Whisper Small had the lowest WER in the evaluated subsets and was selected for the prototype. This selection was based on transcription quality, not a measured speed or memory benchmark.

### Intent and Escalation Classification

The following are **development/validation results**, not scores from an untouched final test set. Baseline and transformer experiments should only be compared with attention to their data splits and averaging methods.

| Task | Approach | Reported result |
|---|---|---|
| Intent | Keyword rules | Micro F1: **0.59** |
| Intent | TF-IDF + Logistic Regression | Micro F1: **0.65** |
| Intent | Fine-tuned CAMeLBERT | Micro F1: **0.81**; Macro F1: **0.763** |
| Escalation (positive class) | Logistic Regression, development run | Precision: **0.19**; Recall: **0.80**; F1: **0.31** |
| Escalation | CAMeLBERT-based classifier | Numerical results require verification against saved evaluation outputs |

For the multi-label intent model, per-label validation thresholds were used: **billing 0.60**, **technical 0.50**, **cancellation 0.35**, **complaint 0.30**, and **information 0.45**. These thresholds reflect validation-set precision–recall/F1 trade-offs, rather than a universal cutoff of 0.50.

In the documented escalation Logistic Regression development run, a **0.30** decision threshold favored recall, but also produced low precision. Final deployment thresholds should be re-evaluated using an independent test set and the actual cost of missed escalations versus false alerts.

### Other Evaluations

- Whisper Small transcription error analysis recorded **2,635 substitutions**, **601 deletions**, and **221 insertions** in the analyzed outputs.
- The grounded digest was compared qualitatively with an LLM-only baseline; quantitative faithfulness/citation-accuracy scores were not reported.
- Full speaker diarization, processing-time benchmarks, and end-to-end quantitative scoring remain areas for further validation.

## Repository Guide

The following are important paths from the team repository; this is **not** an exhaustive or guaranteed current directory tree.

```text
CallSense/
├── app/
│   └── streamlit_app.py               # Shared Streamlit application
├── notebooks/
│   └── fawzia_contribution/           # Classification notebooks + focused README
├── scripts/                            # Training, prediction, evaluation utilities
├── src/                                # Supporting pipeline modules
├── configs/                            # Configuration files, where present
├── requirements.txt                    # Python dependencies
└── README.md                           # This project overview
```

See [`notebooks/fawzia_contribution/README.md`](notebooks/fawzia_contribution/README.md) for the intent/escalation classification contribution and its notebooks.

## Running the Prototype

**Prerequisites:** Python, a working audio-processing setup (including FFmpeg for audio workflows), and access to the model checkpoints and datasets required by the specific pipeline features.

```bash
# 1. Clone the repository
 git clone https://github.com/Fouzex/CallSense.git
 cd CallSense

# 2. Create and activate a virtual environment
 python -m venv .venv
 source .venv/bin/activate

# 3. Install dependencies
 pip install -r requirements.txt

# 4. Launch the Streamlit interface
 streamlit run app/streamlit_app.py
```

> **Important:** These are general startup commands, **not a guarantee of a fully reproducible run from a fresh clone**. Some features require separately prepared audio/transcript data, downloaded model weights, configured paths, or access-controlled model resources. Check the application and configuration files before running. The project was developed and demonstrated in a local environment.

## Responsible AI and Limitations

- The recorded team conversations are **consented simulations** and should not be represented as real customer-service calls.
- Arabic dialect variation, code-switching, and orthographic differences can affect transcription and downstream classification.
- PII redaction is exploratory and should be independently validated before handling sensitive real-world recordings.
- Classification thresholds were selected on development data; independent testing is still needed.
- Retrieval and generated summaries should be checked against source transcripts, especially in high-impact customer-service decisions.
- The prototype is intended to **assist human review**, not automatically make final customer-service or escalation decisions.

## Future Improvements

- Evaluate intent and escalation models on a held-out test set and real-world domain-shifted data.
- Improve Arabic ASR for Saudi dialect and mixed-language speech.
- Validate speaker diarization and PII redaction quantitatively.
- Measure retrieval relevance, digest faithfulness, and citation accuracy.
- Benchmark end-to-end latency, resource use, and deployment reliability.

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Multilingual E5 Base](https://huggingface.co/intfloat/multilingual-e5-base)
- [FAISS](https://github.com/facebookresearch/faiss)
- [MMS Arabic TTS](https://huggingface.co/facebook/mms-tts-ara)
- [Streamlit](https://streamlit.io/)

---

*CallSense is a collaborative AIEP Track E project. Component descriptions and reported metrics summarize the team's documented development experiments.*
