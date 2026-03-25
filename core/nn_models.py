import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification

class RNNForCategoryClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units: int, dropout_rate: float,
                 recurrent_dropout_rate: float, max_sequence_length: int, pad_token_id: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout_rate
        )

        self.fc = nn.Linear(2 * rnn_units, 4)
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embedding(X)
        x = self.dropout(x)
        output, _ = self.rnn(x)

        lengths = (X != self.pad_token_id).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        last_output= output[torch.arange(output.size(0)), lengths - 1, :]

        logits = self.fc(last_output)
        return logits


class CNNForCategoryClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_filters: int, kernel_size: int, dropout_rate: float,
                 max_sequence_length: int, pad_token_id: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)

        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 4)

        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embedding(X)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.global_max_pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits


class LSTMForCategoryClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_units: int, dropout_rate: float,
                 recurrent_dropout_rate: float, max_sequence_length: int, pad_token_id: int = 0, *args, **kwargs):
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout_rate
        )

        self.fc = nn.Linear(2 * lstm_units, 4)
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embedding(X)
        x = self.dropout(x)
        output, _ = self.lstm(x)

        lengths = (X != self.pad_token_id).sum(dim=1)
        lengths = torch.clamp(lengths, min=1)

        last_output= output[torch.arange(output.size(0)), lengths - 1, :]
        logits = self.fc(last_output)
        return logits


class GloveCNNForCategoryClassification(nn.Module):
    def __init__(self, glove_matrix: np.ndarray, num_filters: int, kernel_size: int):
        nn.Module.__init__(self)
        vocab_size, embedding_dim = glove_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(glove_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 4)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.embedding(X)
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = self.global_max_pool(x).squeeze(2)
        logits = self.fc(x)
        return logits


class BertTextClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 4,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        if freeze_encoder:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
