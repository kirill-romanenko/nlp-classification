from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer

from core.nlp import compute_ngram_metrics


class NgramFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k: int = 100):
        self.top_k = top_k
        self.selected_tokens = None

    def fit(self, X: Any, y: Optional[Any] = None):
        if y is not None:
            ngram_metrics = compute_ngram_metrics(texts_tokenized=X, labels=y,
                                                  n=3, metric='anova_f', min_count=50)
            self.selected_tokens = list(ngram_metrics.keys())[:self.top_k]
            return self
        all_tokens = [token for tokens in X if tokens for token in tokens]
        token_counts = Counter(all_tokens)
        self.selected_tokens = [token for token, _ in token_counts.most_common(self.top_k)]
        return self

    def transform(self, X: Any):
        if self.selected_tokens is None:
            return X
        selected = set(self.selected_tokens)
        return [[token for token in (tokens or []) if token in selected] for tokens in X]


class SequenceVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency: int = 1, max_vocab_size: Optional[int] = None,
                 pad_token: str = "<PAD>",
                 oov_token: str = "<OOV>",
                 pad_left: bool = True,
                 max_length: Optional[int] = None,
                 dtype: str = "int32", vocabulary: Optional[dict] = None) -> None:
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.oov_token = oov_token
        self.pad_left = pad_left
        self.max_length = max_length
        self.dtype = dtype
        self.vocabulary = vocabulary

        self.token_to_id_ = None
        self.id_to_token_ = None
        self.pad_token_id_ = 0
        self.oov_token_id_ = 1
        self.vocab_size_ = None
        self.sequence_length_ = None


    def _set_vocabulary(self, vocabulary: dict) -> None:
        self.token_to_id_ = vocabulary
        self.pad_token_id = self.token_to_id_.get(self.pad_token, 0)
        self.oov_token_id = self.token_to_id_.get(self.oov_token, 1)
        max_id = max(self.token_to_id_.values() if self.token_to_id_ else 1)
        id_to_token = [self.oov_token] * (max_id + 1)
        for token, index in self.token_to_id_.items():
            if 0 <= index <= max_id:
                id_to_token[index] = token
        self.id_to_token_ = id_to_token
        self.vocab_size_ = max_id + 1

    def _iter_tokens(self, X: Any):
        for sequence in self._iter_sequences(X):
            for token in sequence:
                if token is not None:
                    yield token

    def _iter_sequences(self, X: Any):
        if X is None:
            return []
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                series = X.iloc[:, 0]
            else:
                candidate_column = None
                for column in X.columns:
                    if len(X[column]) > 0 and isinstance(X[column].iloc[0], (list, tuple)):
                        candidate_column = column
                        break
                if candidate_column is None:
                    raise ValueError("SequenceVectorizer expected a single sequence column. "
                                     "Pass a Series of token lists or a one-column DataFrame.")
                series = X[candidate_column]
            X = series.tolist()
        elif isinstance(X, pd.Series):
            X = X.to_list()
        elif hasattr(X, "to_list") and not isinstance(X, list):
            X = X.tolist()
        for item in X:
            if item is None:
                yield []
            elif isinstance(item, list):
                yield [str(token) for token in item]
            else:
                yield [str(item)]

    def _infer_max_length(self, X: Any) -> int:
        max_length = 0
        vocab = self.token_to_id_ or {}
        for sequence in self._iter_sequences(X):
            filtered_length = 0 if not vocab else sum(1 for token in sequence if token in vocab)
            if filtered_length > max_length:
                max_length = filtered_length
        return max(max_length, 1)

    def fit(self, X: Any, y: Optional[Any] = None):
        if self.vocabulary is not None:
            self._set_vocabulary(self.vocabulary)
        else:
            tokens_iter = self._iter_tokens(X)
            counts = Counter(tokens_iter)

            for special in (self.pad_token, self.oov_token):
                if special in counts:
                    del counts[special]

            items = [(token, count) for token, count in counts.items() if count >= self.min_frequency]
            items.sort(key=lambda x: (-x[1], x[0]))

            if self.max_vocab_size is not None:
                max_regular = max(0, self.max_vocab_size - 2)
                items = items[:max_regular]

            vocab = {self.pad_token: self.pad_token_id_, self.oov_token: self.oov_token_id_}
            for index, (token, count) in enumerate(items, start=2):
                vocab[token] = index
            self._set_vocabulary(vocab)
        if self.max_length is not None:
            self.sequence_length_ = self.max_length
            return self
        self.sequence_length_ = self._infer_max_length(X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.token_to_id_ is None or self.sequence_length_ is None:
            raise ValueError("Vectorizer must be fitted before transform")

        sequences = []
        to_id = self.token_to_id_
        oov_id = self.oov_token_id_
        max_length = self.sequence_length_

        for tokens in self._iter_sequences(X):
            filtered_tokens = [token for token in tokens if token in to_id]
            sequence = [to_id[token] for token in filtered_tokens]

            if len(sequence) > max_length:
                if self.pad_left:
                    sequence = sequence[-max_length:]
                else:
                    sequence = sequence[:max_length]

            pad_needed = max_length - len(sequence)
            if pad_needed > 0:
                pad_chunk = [self.pad_token_id_] * pad_needed
                sequence = pad_chunk + sequence if self.pad_left else sequence + pad_chunk

            sequences.append(sequence)

        return np.asarray(sequences, dtype=self.dtype)

    def inverse_transform(self, X: Any) -> list[list[str]]:
        if self.id_to_token_ is None:
            raise ValueError("Vectorizer must be fitted before inverse_transform")
        id_to_token = self.id_to_token_
        result = []

        for row in np.asarray(X):
            result.append([id_to_token[i] if 0 <= i < len(id_to_token) else self.oov_token for i in row])
        return result

    def get_vocabulary(self) -> dict:
        if self.token_to_id_ is None:
            raise ValueError("Vectorizer must be fitted before getting vocabulary")
        return dict(self.token_to_id_)


class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str, max_length: Optional[int] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model_name = model_name

    def fit(self, X: Any, y: Optional[Any] = None):
        return self.transform(X)

    def transform(self, texts: Any) -> np.ndarray:
        encoded = self.tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np"
        )
        return encoded["input_ids"]
