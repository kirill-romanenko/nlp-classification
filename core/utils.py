import gc
import random

import numpy as np
import pandas as pd
import torch
from torch import nn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    # torch.mps.empty_cache()


def divide_data(data: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_glove_fixed_vocab(path: str, embedding_dim: int, max_words: int) -> tuple[dict, np.ndarray]:
    vocab = {'<PAD>': 0, '<OOV>': 1}
    matrix = [np.zeros(embedding_dim, dtype=np.float32), np.zeros(embedding_dim, dtype=np.float32)]
    added = 0
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if added > max_words:
                break
            parts = line.strip().split(' ')
            if len(parts) < embedding_dim + 1:
                continue
            token = parts[0]
            if token in vocab:
                continue
            vector = np.asarray(parts[1:], dtype=np.float32)
            if vector.shape[0] != embedding_dim:
                continue
            idx = len(vocab)
            vocab[token] = idx
            matrix.append(vector)
            added += 1
    embedding_matrix = np.vstack(matrix)
    return vocab, embedding_matrix


def build_submission_dataframe(test_data: pd.DataFrame, id_column: str,
                               target_column: str, predictions: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame()
    result[id_column] = test_data[id_column]
    result[target_column] = predictions
    return result
