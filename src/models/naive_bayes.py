import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import logsumexp


class MultinomialNaiveBayes:
    """
    Binary Multinomial Naive Bayes implemented from scratch.
    
    Input:
        X_counts : sparse matrix (n_samples, n_features)
        y        : binary labels (0 or 1)
        
    Output:
        predict_proba -> probability of class 1
        predict       -> predicted class (0 or 1)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.class_log_prior_ = None      # log P(class)
        self.feature_log_prob_ = None     # log P(word | class)

    def fit(self, X_counts, y):
        y = np.asarray(y).astype(int)
        n_docs, n_features = X_counts.shape

        # Compute class prior probabilities
        class_counts = np.bincount(y, minlength=2)
        self.class_log_prior_ = np.log(class_counts / n_docs)

        # Count word occurrences per class
        word_counts = np.zeros((2, n_features), dtype=np.float64)
        for c in (0, 1):
            Xc = X_counts[y == c]
            word_counts[c] = np.asarray(Xc.sum(axis=0)).ravel()

        # Laplace smoothing
        smoothed_wc = word_counts + self.alpha
        smoothed_tot = smoothed_wc.sum(axis=1, keepdims=True)

        # log P(word | class)
        self.feature_log_prob_ = np.log(smoothed_wc) - np.log(smoothed_tot)

        return self

    def _joint_log_likelihood(self, X_counts):
        # log P(class) + sum log P(words | class)
        return X_counts @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_proba(self, X_counts):
        # Compute normalized probability of class 1
        jll = self._joint_log_likelihood(X_counts)
        log_norm = logsumexp(jll, axis=1)
        return np.exp(jll[:, 1] - log_norm)

    def predict(self, X_counts, threshold=0.5):
        return (self.predict_proba(X_counts) >= threshold).astype(int)


def fit_multilabel_nb(df_train, df_valid, text_col: str, labels: list[str],
                      alpha=1.0, min_df=5, max_df=0.95):
    """
    Train one Naive Bayes classifier per label (multi-label setting).

    Returns:
        vectorizer : CountVectorizer
        models     : dict[label -> MultinomialNaiveBayes]
        X_valid_counts : vectorized validation texts
    """

    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)

    X_train = vectorizer.fit_transform(df_train[text_col])
    X_valid = vectorizer.transform(df_valid[text_col])

    models = {}
    for label in labels:
        nb = MultinomialNaiveBayes(alpha=alpha)
        nb.fit(X_train, df_train[label].values)
        models[label] = nb

    return vectorizer, models, X_valid
