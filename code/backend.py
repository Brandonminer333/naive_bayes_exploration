import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def make_synthetic_data(
    n: int,
    w: int,
    c: int,
    avg_doc_length: int = 50,
    class_sep: float = 1.0,
    random_state: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic document-word data suitable for Naive Bayes experiments.

    Parameters
    ----------
    n : int
        Number of samples (documents).
    w : int
        Number of columns (words in vocabulary).
    c : int
        Number of classes.
    avg_doc_length : int, optional
        Expected number of words per document (Poisson-distributed).
    class_sep : float, optional
        Controls how different the class word distributions are.
        Higher = more separable classes. Lower = more overlap.
        Typically between 0.1 and 3.0.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Document-word matrix of shape (n, w) containing word counts.
    y : np.ndarray
        Class labels of length n.
    """
    rng = np.random.default_rng(random_state)

    # --- 1. Create base word distribution (Dirichlet) ---
    base_word_dist = rng.dirichlet(np.ones(w))

    # --- 2. Create per-class word distributions ---
    # Pull the class distributions toward the base distribution
    class_word_dists = []
    for _ in range(c):
        alpha = base_word_dist * (1.0 / class_sep)
        dist_k = rng.dirichlet(alpha)
        class_word_dists.append(dist_k)
    class_word_dists = np.vstack(class_word_dists)

    # --- 3. Sample classes ---
    y = rng.integers(low=0, high=c, size=n)

    # --- 4. Generate documents for each sample ---
    X = np.zeros((n, w), dtype=int)
    for i in range(n):
        doc_len = rng.poisson(avg_doc_length)
        class_dist = class_word_dists[y[i]]
        X[i] = rng.multinomial(doc_len, class_dist)

    return X, y


def synthetic_model(X, y, random_state: str = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def synthetic_confusion(X, y, random_state: str = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return confusion_matrix(y_test, preds)
