# Mental Health Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JTunnessen/mental_health_analysis/blob/main/v2_Mental_Health_Sentiment_Analysis.ipynb)

A multi-class text classifier that categorises mental health statements into diagnostic categories using a bidirectional-style 2-layer LSTM with an attention mechanism and pre-trained GloVe word embeddings.

---

## Overview

Mental health screening at scale is constrained by the availability of trained clinicians. This project explores whether a lightweight RNN-based model can reliably classify free-text statements into clinically relevant sentiment/status categories (e.g. *Depression*, *Anxiety*, *Suicidal*, *Normal*, etc.), and provides token-level attention heatmaps to make predictions interpretable.

The training pipeline uses two phases:
1. **Phase 1** — frozen GloVe embeddings, higher learning rate (Adam, lr=0.001)
2. **Phase 2 (fine-tuning)** — embeddings unfrozen, lower learning rate (Adam, lr=0.0001)

Early stopping is applied in both phases to prevent overfitting.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) (Kaggle) |
| File | `Combined Data.csv` |
| Input column | `statement` |
| Target column | `status` |
| Split | 70% train / 15% val / 15% test (stratified) |

The dataset contains real-world social-media and forum text labelled with mental health conditions. The number of target classes and exact class distribution are printed when the notebook is executed.

---

## Architecture

```
Input text
    │
    ▼
Embedding layer  (300-d GloVe vectors, frozen in Phase 1)
    │
    ▼
2-layer LSTM  (batch-first, dropout=0.5 between layers)
    │
    ▼
Attention mechanism  (linear → softmax → weighted sum)
    │
    ▼
Dropout (0.5)
    │
    ▼
Fully-connected layer → softmax → predicted class
```

**Key hyperparameters**

| Parameter | Value |
|---|---|
| Embedding dimension | 300 |
| LSTM hidden dimension | configurable (`HIDDEN_DIM`) |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Phase 1 optimizer | Adam, lr=0.001 |
| Phase 2 optimizer | Adam, lr=0.0001, weight_decay=1e-5 |
| Max gradient norm | configurable (`MAX_GRAD_NORM`) |
| Early stopping patience | 3 epochs |

---

## Features

- **Multi-class classification** — handles all status categories present in the dataset
- **GloVe embeddings** — 300-dimensional pre-trained vectors (dolma 2024 release); falls back to random initialisation if unavailable
- **Two-phase training** — frozen then fine-tuned embeddings for improved generalisation
- **Explainable AI** — token-level attention heatmaps via `plot_attention()` show which words drive each prediction
- **Device-agnostic** — automatically selects CUDA, Apple MPS, or CPU

---

## Setup & Installation

### Requirements

```
Python >= 3.9
torch >= 2.0
pandas
numpy
scikit-learn
matplotlib
seaborn
kagglehub
```

Install with pip:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn kagglehub
```

### Kaggle credentials

`kagglehub` requires a Kaggle API token (`~/.kaggle/kaggle.json`). Obtain one from your [Kaggle account settings](https://www.kaggle.com/settings) and place it at the path above, or set the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.

### GloVe embeddings

The notebook downloads GloVe vectors automatically from Stanford NLP. If the download fails (e.g. network restrictions in Colab), the `simulate_glove_vectors()` fallback is used instead — training will still work but may produce lower accuracy.

---

## Usage

### Running in Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this README.
2. Ensure your Kaggle credentials are available (mount Google Drive or set env vars).
3. Run all cells in order (`Runtime → Run all`).

### Running locally

```bash
git clone https://github.com/JTunnessen/mental_health_analysis.git
cd mental_health_analysis
pip install torch pandas numpy scikit-learn matplotlib seaborn kagglehub
jupyter notebook v2_Mental_Health_Sentiment_Analysis.ipynb
```

Run cells top to bottom. Training takes several minutes on CPU; a GPU is recommended.

### Single-sentence inference

After the notebook has been run, use `predict_sentiment()`:

```python
label, confidence = predict_sentiment(
    "I feel completely hopeless and I don't want to continue.",
    model,
    word_to_index,
    index_to_label
)
print(f"{label}  ({confidence:.2%})")
```

### Attention visualisation

```python
plot_attention(
    "I can't sleep because I'm so worried about everything.",
    viz_model,
    word_to_index,
    index_to_label
)
```

This renders a heatmap showing the attention weight the model assigned to each token.

---

## Project Structure

```
mental_health_analysis/
└── v2_Mental_Health_Sentiment_Analysis.ipynb   # Main notebook
```

### Notebook cell overview

| Cells | Purpose |
|---|---|
| 0–1 | Colab badge, title |
| 2 | All imports + `evaluate_model`, `print_metrics`, `plot_confusion_matrix` helpers |
| 3 | Data loading (kagglehub download, train/val/test split) |
| 4 | Text cleaning (`clean_text_v2`), vocabulary, sequence padding, tensors |
| 5 | GloVe utilities, `Attention`, `EarlyStopping`, `ClassificationRNN`, Phase 1 training |
| 6–9 | Phase 1 evaluation: predictions, accuracy, classification report, confusion matrix |
| 10–14 | Phase 2 fine-tuning + evaluation |
| 15–16 | Real-world inference (`predict_sentiment`) |
| 17–19 | Explainable AI: `ClassificationRNN_Viz`, `plot_attention` |

---

## Model Performance

Evaluation results are printed to the notebook output after each training phase. Look for:

- **Overall Accuracy** — single-number summary
- **Classification Report** — per-class precision, recall, and F1-score
- **Confusion Matrix** — heatmap showing prediction distribution across classes

Phase 2 results are labelled **(After Fine-Tuning)**.

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
