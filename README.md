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


---
## Reproducibility & “do not rerun everything”

To avoid re-running heavy preprocessing and overnight training:

### 1) Processed data is stored locally (if present)
You **do not need** to re-run the full preprocessing pipeline, the processed splits are in `data/processed/`
Notebooks will directly reuse the processed files.

- `data/raw/` is optional cache (not versioned)
- `data/processed/` contains processed datasets/splits 

> If `data/processed/` is missing on your machine, you need to run the notebook `2_preprocessing.ipynb`
.

### 2) Lightweight Transformer: weights + vocabulary are saved
For the from-scratch Transformer, the final model weights and the associated vocabulary are saved under:

- `models/`
Same for xgboost and naive bayes models, we couldn't uopload the finetuned DistilBert on Github since it's heavy

So you can **evaluate / reproduce metrics without retraining overnight**.


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
├── raw/                # raw cache 
└── processed/          # processed datasets/splits (recommended to keep locally)

notebooks/
├── 1_EDA.ipynb
├── 2_preprocessing.ipynb
├── 3_naive_bayes_classification.ipynb
├── 5_distilbert_finetuning.ipynb
└── 6_simple_transformer_classification.ipynb

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
Run:  
notebooks/1_EDA.ipynb  

Produces dataset statistics and class imbalance visualizations.

2) Preprocessing Experiments  
Run:  
notebooks/2_preprocessing.ipynb  

Compares:
- Light preprocessing (xLSTM-style)  
- Aggressive preprocessing

3) Naive Bayes baseline (from scratch)  
Run:  
notebooks/3_naive_bayes_classification.ipynb  

Uses:
- src/models/naive_bayes.py  
- CountVectorizer word-count features  
- One-vs-rest training per label

4) TF-IDF + XGBoost  
Run:  
notebooks/4_tfidf_xgboost_classification.ipynb  
- Train one XGBoost model per label  
- Thresholds selected on validation set by maximizing F1-score

5) DistilBERT fine-tuning  
Run:  
notebooks/5_distilbert_finetuning.ipynb  

Training is slow on CPU. GPU recommended if available.

6) Lightweight Transformer (from scratch)  
Run:  
notebooks/6_simple_transformer_classification.ipynb  

If saved weights exist in models/, evaluation runs directly.  
Otherwise, training runs from scratch (long on CPU).

## Notes on Reproducibility
Virtual environments (.venv) are not versioned.

Raw and processed datasets are excluded from version control.

All experiments can be reproduced using the provided notebooks and requirements.txt.

## Authors
Zakarya ALAMI DRIDEB

Wiam LACHQER
