# Toxic-Comment-Classification-using-Classical-NLP-Methods-and-a-Lightweight-Transformer
This repository contains the code and resources for the Advanced Machine Learning project by Wiam Lachqer. and Zakarya Alami Drideb. The project focuses on Toxic Comment Classification using classical NLP methods and a lightweight transformer model.

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

## Dataset
The experiments are conducted on the Civil Comments dataset (Google / Jigsaw).

The dataset is automatically downloaded using the HuggingFace datasets library.
No manual download of CSV files is required.

## Project Structure
```bash
1) data/
├── raw/                # raw data cache (optional, not versioned)
├── processed/          # processed datasets (not versioned)

2) notebooks/
├── EDA.ipynb
├── preprocessing.ipynb
├── model_baselines.ipynb
├──preprocessing_draft_from_repo.ipynb

3) src/
├── preprocessing.py    # text cleaning and normalization
├── models/
│   ├── naive_bayes.py
│   └── transformer.py
├── utils.py

4) models/                 # trained models (not versioned)
reports/
├── figures/
└── report.pdf
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
