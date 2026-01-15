""" This module implements the text preprocessing pipeline. 
    It defines all the necessary functions, classes, and objects, and implements all
    the necessary preprocessing steps in the preprocess() function at the bottom.
"""

import re
from unidecode import unidecode
import contractions



################# Helpers + pipeline xLSTM light


import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
_PUNCTUATION = re.compile(r"[^\w\s]")
_DUPE_CHARS  = {r"([A-Za-z])\1{2,}"          : r"\1\1"}  # Deduplication of multiple consecutive sequences of consecutive duplicate characters 
                                                        # in a string. eg: cooool -> cool, goooaaal -> goal
_HTML        = {r'<.*?>'}    # Removal of html tags

def _lower(text: str) -> str:
    return text.lower()

def _remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)

def _remove_emails(text: str) -> str:
    return _EMAIL_RE.sub(" ", text)

def _replace_numbers(text: str, token: str = "<NUM>") -> str:
    return _NUM_RE.sub(f" {token} ", text)

def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _spacy_lemma(text: str, nlp) -> str:
    doc = nlp(text)
    return " ".join(tok.lemma_ for tok in doc if not tok.is_space)

def _remove_stopwords(text: str, stopwords: set[str]) -> str:
    toks = text.split()
    return " ".join(t for t in toks if t not in stopwords)

def preprocess_light_xlstm(text: str, *, nlp, stopwords: set[str]) -> str:
    """
    xLSTM-inspired light preprocessing:
    lowercase -> remove URLs/emails -> replace numbers -> lemmatize -> stopwords -> whitespace normalize
    """
    if text is None:
        return ""
    x = text
    x = _lower(x)
    x = _remove_urls(x)
    x = _remove_emails(x)
    x = _replace_numbers(x, token="<NUM>")
    # x = _norm_ws(x)
    x = _spacy_lemma(x, nlp)
    x = _remove_stopwords(x, stopwords)
    x = _norm_ws(x)
    return x


def preprocess_light_xlstm_from_doc(doc, stopwords: set[str]) -> str:
    if doc is None:
        return ""

    x = " ".join(tok.text for tok in doc)
    x = _lower(x)
    x = _remove_urls(x)
    x = _remove_emails(x)
    x = _replace_numbers(x, token="<NUM>")
    # x = _norm_ws(x)
    x = " ".join(tok.lemma_ for tok in doc if not tok.is_space) # lemmatize
    x = _remove_stopwords(x, stopwords)
    x = _norm_ws(x)

    return x



##### Specific functions for aggressive processing

def _remove_punctuation(text: str) -> str:
    return _PUNCTUATION.sub(" ", text)

def _remove_duplicates(text: str) -> str:
    for pattern, repl in _DUPE_CHARS.items():
        text = re.sub(pattern, repl, text)
    return text

def _remove_html(text: str) -> str:
    for pattern in _HTML:
        text = re.sub(pattern, " ", text)
    return text

def preprocess(text: str, *, nlp, stopwords: set[str]) -> str:
    """
    xLSTM-inspired light preprocessing:
    lowercase -> remove URLs/emails -> replace numbers -> unidecode -> HTML artifacts -> expand contractions-> lemmatize -> stopwords ->  punctuation -> whitespace normalize
    """
    if text is None:
        return ""
    x = text
    x = _lower(x)    # 1
    x = _remove_urls(x)  # 3
    x = _remove_emails(x)   # 3 
    x = _replace_numbers(x, token="<NUM>")  # 3
    x = unidecode(x)
    x = _remove_html(x)
    x = _remove_duplicates(x)
    x = contractions.fix(x)    # Expand contractions
    # x = _norm_ws(x)
    x = _spacy_lemma(x, nlp)  # 2
    x = _remove_stopwords(x, stopwords)  # 4
    x = _remove_punctuation(x)   # Remove punctuation
    x = _norm_ws(x) 
    return x

def preprocess_from_doc(doc, stopwords: set[str]) -> str:
    if doc is None:
        return ""

    x = " ".join(tok.text for tok in doc)
    x = _lower(x)
    x = _remove_urls(x)
    x = _remove_emails(x)
    x = _replace_numbers(x, token="<NUM>")
    x = unidecode(x)
    x = _remove_html(x)
    x = contractions.fix(x)
    x = _remove_duplicates(x)
    # x = _norm_ws(x)
    x = " ".join(tok.lemma_ for tok in doc if not tok.is_space)
    x = _remove_stopwords(x, stopwords)
    x = _remove_punctuation(x)
    x = _norm_ws(x)

    return x