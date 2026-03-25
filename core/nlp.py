import string
from collections import Counter
from typing import Sequence, Any

import nltk
import numpy as np
import pandas as pd
import tiktoken
import ssl

from nltk.corpus import stopwords
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

TIKTOKEN_ENCODING = tiktoken.get_encoding('cl100k_base')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('stopwords')


def tokenize_tiktoken(text: str) -> list[str]:
    token_bytes = [
        TIKTOKEN_ENCODING.decode_single_token_bytes(token)
        for token in TIKTOKEN_ENCODING.encode(text)
    ]

    tokens = [token.decode('utf-8', errors='replace').strip().lower() for token in token_bytes]
    return [token for token in tokens if token]


def get_stopwords_set(language: str = 'english', custom_stopwords: set[str] = None) \
        -> set[str]:
    default_stopwords = set(stopwords.words(language))
    if custom_stopwords:
        return default_stopwords.union(custom_stopwords)
    return default_stopwords


def _get_punctuation_set(custom_punctuation: set[str] = None) -> set[str]:
    default_punctuation = set(string.punctuation)
    if custom_punctuation:
        return default_punctuation.union(custom_punctuation)
    return default_punctuation

def _is_punctuation_token(token: str, custom_punctuation: set[str] = None) -> bool:
    punctuation_set = _get_punctuation_set(custom_punctuation)
    stripped = token.strip()
    return stripped in punctuation_set or (stripped != '' and all(c in punctuation_set for c in stripped))


def filter_tokens(
        tokens: Sequence[str],
        *,
        remove_stopwords: bool = False,
        remove_punctuation_tokens: bool = False,
        custom_stopwords: set[str] = None,
        custom_punctuation: set[str] = None,
        lowercase_for_counting: bool = False,
        language: str = 'english',
) -> list[str]:
    stopwords = get_stopwords_set(language, custom_stopwords) if remove_stopwords else None
    result = []

    for token in tokens:
        t = str(token).strip()
        if not t:
            continue
        if remove_punctuation_tokens and _is_punctuation_token(token, custom_punctuation):
            continue
        if stopwords and t.lower() in stopwords:
            continue

        final_token = t.lower() if lowercase_for_counting else t
        result.append(final_token)
    return result


def _flatten_array_column(df: pd.DataFrame, column_name: str) -> list[Any]:
    return [item for sublist in df[column_name] for item in sublist]


def _value_counts(values: Sequence[Any]) -> pd.Series:
    return pd.Series(list(values)).value_counts()


def _generate_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def generate_all_ngrams(tokens: list[str], max_n: int) -> list[tuple[str, ...]]:
    if max_n < 1:
        raise ValueError("max_n must be >= 1")

    all_ngrams = []
    for i in range(1, max_n + 1):
        all_ngrams.extend(_generate_ngrams(tokens, i))
    return all_ngrams


def punctuation_counts(df: pd.DataFrame, column: str,
                       custom_punctuation: set[str] = None) -> pd.Series:
    all_tokens = _flatten_array_column(df, column)
    punctuation = [token for token in all_tokens if _is_punctuation_token(str(token), custom_punctuation)]
    return _value_counts(punctuation)


def stopwords_counts(df: pd.DataFrame, column: str, language: str = 'english',
                     custom_stopwords: set[str] = None) -> pd.Series:
    all_tokens = _flatten_array_column(df, column)
    stop_words = get_stopwords_set(language, custom_stopwords)
    stopwords_in_series = [token for token in all_tokens if str(token).strip().lower() in stop_words]
    return _value_counts(stopwords_in_series)


def token_counts(df: pd.DataFrame, column: str, remove_stopwords: bool = False,
                 remove_punctuation: bool = False,
                 custom_stopwords: set[str] = None,
                 custom_punctuation: set[str] = None,
                 lowercase_for_counting: bool = False) -> pd.Series:
    all_tokens = _flatten_array_column(df, column)
    processed = filter_tokens(
        [str(token) for token in all_tokens],
        remove_stopwords=remove_stopwords,
        remove_punctuation_tokens=remove_punctuation,
        custom_stopwords=custom_stopwords,
        custom_punctuation=custom_punctuation,
        lowercase_for_counting=lowercase_for_counting
    )
    return _value_counts(processed)


def compute_ngram_metrics(
        texts_tokenized: list[list[str]],
        labels: list[int],
        n: int = 1,
        metric: str = 'anova_f',
        min_count: int = 1) -> dict[str, float]:
    if len(texts_tokenized) != len(labels):
        raise ValueError("texts and labels must have same length")
    if n < 1:
        raise ValueError("n must be >= 1")
    if metric not in ['anova_f', 'mutual_info']:
        raise ValueError("metric must be 'anova_f' or 'mutual_info'")

    unique_classes = sorted(set(labels))
    ngrams_by_class = {cls: [] for cls in unique_classes}

    for tokens, label in zip(texts_tokenized, labels):
        all_ngrams = generate_all_ngrams(tokens, n)
        ngrams_by_class[label].extend(all_ngrams)

    frequency_by_class = {cls: Counter(ngrams_by_class[cls]) for cls in unique_classes}

    all_ngrams = set()
    for frequency_dict in frequency_by_class.values():
        all_ngrams.update(frequency_dict.keys())

    if min_count > 1:
        total_counts = Counter()
        for frequency_dict in frequency_by_class.values():
            total_counts.update(frequency_dict)

        filtered_ngrams = {ngram for ngram in all_ngrams if total_counts[ngram] >= min_count}

        all_ngrams = filtered_ngrams

        for cls in unique_classes:
            frequency_by_class[cls] = {ngram: count for ngram, count in frequency_by_class[cls].items()
                                  if ngram in filtered_ngrams}

    vocab = list(all_ngrams)
    vocab_to_idx = {ngram: i for i, ngram in enumerate(vocab)}

    X = np.zeros((len(texts_tokenized), len(vocab)))

    for doc_idx, tokens in enumerate(texts_tokenized):
        doc_ngrams = generate_all_ngrams(tokens, n)
        doc_ngram_counts = Counter(doc_ngrams)

        for ngram, count in doc_ngram_counts.items():
            if ngram in vocab_to_idx:
                X[doc_idx, vocab_to_idx[ngram]] = count

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    match metric:
        case 'anova_f':
            f_scores, p_values = f_classif(X, y_encoded)
            metric_values = f_scores
        case 'mutual_info':
            mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
            metric_values = mi_scores

    ngram_to_string = {ngram: ' '.join(ngram) for ngram in all_ngrams}
    metric_dict = {}
    for i, ngram in enumerate(vocab):
        token_str = ngram_to_string[ngram]
        metric_dict[token_str] = metric_values[i]

    return dict(sorted(metric_dict.items(), key=lambda item: item[1], reverse=True))


def count_based_analysis(
        texts_tokenized: list[list[str]],
        labels: list[int],
        n: int = 1,
        metric: str = 'anova_f',
        min_count: int = 1) -> pd.DataFrame:
    metric_dict = compute_ngram_metrics(texts_tokenized, labels, n, metric, min_count)
    unique_classes = sorted(set(labels))

    ngrams_by_class = {cls: [] for cls in unique_classes}
    for tokens, label in zip(texts_tokenized, labels):
        all_ngrams = generate_all_ngrams(tokens, n)
        ngrams_by_class[label].extend(all_ngrams)

    frequency_by_class = {cls: Counter(ngrams_by_class[cls]) for cls in unique_classes}

    if min_count > 1:
        total_counts = Counter()
        for frequency_dict in frequency_by_class.values():
            total_counts.update(frequency_dict)

        filtered_ngrams = {ngram for ngram in total_counts if total_counts[ngram] >= min_count}

        for cls in unique_classes:
            frequency_by_class[cls] = {ngram: count for ngram, count in frequency_by_class[cls].items()
                                       if ngram in filtered_ngrams}

    df = pd.DataFrame([
        {'token': token, 'metric': metric_value}
        for token, metric_value in metric_dict.items()
    ])

    for cls in unique_classes:
        count_col = f'count_{cls}'
        freq_col = f'freq_{cls}'

        ngram_to_string = {}
        for ngram in frequency_by_class[cls].keys():
            ngram_to_string[' '.join(ngram)] = frequency_by_class[cls][ngram]

        df = df.merge(
            pd.Series(ngram_to_string, name=count_col).reset_index().rename(columns={'index': 'token'}),
            on='token',
            how='left'
        ).fillna(0)

        df[count_col] = df[count_col].astype(int)

        total_ngrams = len(ngrams_by_class[cls])
        df[freq_col] = df[count_col] / total_ngrams if total_ngrams > 0 else 0

    count_cols = [f'count_{cls}' for cls in unique_classes]
    df['total_count'] = df[count_cols].sum(axis=1)

    return df.reset_index(drop=True)
