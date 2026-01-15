# Toxic-Comment-Classification-using-Classical-NLP-Methods-and-a-Lightweight-Transformer
This repository contains the code and resources for the Advanced Machine Learning project by Wiam Lachqer, and Zakarya Alami Drideb. The project focuses on Toxic Comment Classification using classical NLP methods and a lightweight transformer model.
We compare:
- Classical baselines: **Multinomial Naive Bayes (from scratch)**, **TF-IDF + XGBoost**
- Neural models: **fine-tuned DistilBERT** and a **lightweight Transformer encoder (from scratch)**

The task is **multi-label classification** over 6 toxicity categories.

---
### Dataset used for the final experiments
The final experiments are conducted on the **binarized multilabel dataset**:

**`jigsaw-toxic-comment-classification-challenge`**  
(6 binary labels: toxic, severe_toxicity, obscene, threat, insult, identity_attack)

This is the dataset used in the report and in the main training/evaluation notebooks.
---
## Reproducibility & “do not rerun everything”

To avoid re-running heavy preprocessing and overnight training:

### 1) Processed data is stored locally (if present)
If you already have the processed splits in `data/processed/`, you **do not need** to re-run the full preprocessing pipeline.
Notebooks will directly reuse the processed files (when available).

- `data/raw/` is optional cache (not versioned)
- `data/processed/` contains processed datasets/splits (recommended to keep locally)

> If `data/processed/` is missing on your machine, notebooks will re-download and re-process from the dataset.

### 2) Lightweight Transformer: weights + vocabulary are saved
For the from-scratch Transformer, the final model weights and the associated vocabulary are saved under:

- `models/`

So you can **evaluate / reproduce metrics without retraining overnight**.

> If `models/` files are missing, the notebook can retrain the model, but this is slow on CPU.

---
## Installation

### Prerequisites
- **Python ≥ 3.10** (tested with Python 3.11)
- **Windows users**: PyTorch may require the installation of the *Microsoft Visual C++ Redistributable (2015–2022)*  
  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

---

### 1) Create a virtual environment
```bash
python -m venv .venv
```
# Windows PowerShell
```bash
.\.venv\Scripts\Activate.ps1

### 2) Install PyTorch (CPU version) and spacy 
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
The CPU version is intentionally used to ensure reproducibility across all machines.
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt wordnet averaged_perceptron_tagger
```

### 3) Install remaining dependencies
```bash
pip install -r requirements.txt
```


## Project Structure
```bash
data/
├── raw/                # raw cache (optional)
└── processed/          # processed datasets/splits (recommended to keep locally)

notebooks/
├── EDA_xlstm table.ipynb
├── preprocessing_xlstm_table.ipynb
├── naive_bayes_classification.ipynb
├── simple_transformer_classification.ipynb
└── distilbert_finetuning.ipynb

src/
├── __init__.py
├── preprocessing.py
└── models/
    ├── naive_bayes.py              # Multinomial NB from scratch
    └── simple_transformer.py       # Transformer encoder from scratch

models/
└── (saved weights + vocabulary for lightweight transformer, if provided)
```

## Recommended Workflow
1) Exploratory Data Analysis
notebooks/data_exploration.ipynb

2) Preprocessing Experiments
Comparison between light and aggressive text cleaning strategies
notebooks/preprocessing.ipynb

3) Baseline Models
TF-IDF + linear models, Naive Bayes
notebooks/model_baselines.ipynb

4) Transformer Model
Lightweight Transformer encoder implemented and evaluated incrementally

## Notes on Reproducibility
Virtual environments (.venv) are not versioned.

Raw and processed datasets are excluded from version control.

All experiments can be reproduced using the provided notebooks and requirements.txt.

## Authors
Zakarya ALAMI DRIDEB

Wiam LACHQER
