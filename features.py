from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """
    Builds a scikit-learn Pipeline with TF-IDF + Logistic Regression.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
        stop_words="english",
        strip_accents="unicode",
        sublinear_tf=True,
    )

    # 'saga' handles large sparse data well
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=None  # LogisticRegression doesn't accept n_jobs; leave as None
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
