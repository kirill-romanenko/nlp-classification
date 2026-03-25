import copy
from time import perf_counter
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from sklearn import clone
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import core.visualization as p
from core.baseline_training import (
    aggregate_classification_cv_metrics,
    evaluate_classification,
)
from core.training_results import ClassificationMetrics
from core.utils import free_memory, get_device


def _safe_index(data: Any, indices: Union[Sequence[int], np.ndarray]) -> Any:
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    try:
        return data.iloc[indices]
    except Exception:
        return np.asarray(data)[indices]


def _labels_from_score(y_score: np.ndarray, threshold: float) -> np.ndarray:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return (y_score.ravel() >= threshold).astype(int)
    return np.argmax(y_score, axis=1)


def _positive_class_probabilities(y_score: np.ndarray) -> Optional[np.ndarray]:
    if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
        return y_score.ravel()
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        return y_score[:, 1]
    return None


def cross_validate_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv,
    criterion: nn.Module,
    optimizer_class: type,
    optimizer_params: dict,
    preprocessor: Optional[Any] = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: Optional[str] = None,
    enable_plot: Optional[bool] = False,
    model_name: Optional[str] = "PyTorch CV",
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 1e-4,
    validation_size: float = 0.1,
    restore_best_weights: bool = True,
) -> ClassificationMetrics:
    device = torch.device(device or get_device())

    fit_times, accs, f1s, precs, recalls, aucs = [], [], [], [], [], []
    estimators = []

    X = np.array(X)
    y = np.array(y)

    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    oof_pred = np.empty(n_samples, dtype=int)
    oof_probs = np.empty((n_samples, n_classes), dtype=float)
    has_probs = True

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\n🌀 Fold {fold_index + 1}/{cv.get_n_splits()}")

        X_train_full, y_train_full = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_size,
            random_state=42,
            stratify=y_train_full,
        )

        fitted_preprocessor = clone(preprocessor)
        if fitted_preprocessor is not None:
            X_train = fitted_preprocessor.fit_transform(X_train)
            X_val = fitted_preprocessor.transform(X_val)
            X_test = fitted_preprocessor.transform(X_test)

        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        X_val_tensor = torch.tensor(X_val, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

        fold_model = copy.deepcopy(model).to(device)
        optimizer = optimizer_class(fold_model.parameters(), **optimizer_params)

        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = 0
        epochs_without_improvement = 0

        start_fit = perf_counter()

        for epoch in tqdm(
            range(num_epochs), desc=f"Training fold {fold_index + 1}", leave=True
        ):
            # ---- train ----
            fold_model.train()
            total_train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = fold_model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                del X_batch, y_batch, logits, loss

            avg_train_loss = total_train_loss / len(train_loader)

            # ---- validation ----
            fold_model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    val_logits = fold_model(X_batch)
                    val_loss = criterion(val_logits, y_batch)
                    total_val_loss += val_loss.item()

                    del X_batch, y_batch, val_logits, val_loss

            avg_val_loss = total_val_loss / len(val_loader)

            tqdm.write(
                f"  Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            # ---- early stopping ----
            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                if restore_best_weights:
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in fold_model.state_dict().items()
                    }
            else:
                epochs_without_improvement += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if epochs_without_improvement >= early_stopping_patience:
                tqdm.write(
                    f"  Early stopping on epoch {epoch + 1} "
                    f"(best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})"
                )
                break

        if restore_best_weights and best_state_dict is not None:
            fold_model.load_state_dict(best_state_dict)
            fold_model.to(device)

        fit_times.append(perf_counter() - start_fit)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- test inference in batches ----
        fold_model.eval()
        all_probs = []

        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)

                logits = fold_model(X_batch)
                probs = torch.softmax(logits, dim=1).cpu()
                all_probs.append(probs)

                del X_batch, logits, probs

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        y_probs = torch.cat(all_probs, dim=0).numpy()
        y_pred = np.argmax(y_probs, axis=1)

        fold_metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            model_name=f"fold_{fold_index + 1}",
            enable_plot=enable_plot,
        )

        accs.append(fold_metrics.accuracy)
        f1s.append(fold_metrics.f1_score)
        precs.append(fold_metrics.precision)
        recalls.append(fold_metrics.recall)
        aucs.append(
            fold_metrics.roc_auc if fold_metrics.roc_auc is not None else float("nan")
        )

        estimators.append(
            {
                "preprocessor": fitted_preprocessor,
                "model": copy.deepcopy(fold_model).cpu(),
            }
        )

        oof_pred[test_index] = y_pred
        oof_probs[test_index] = y_probs if y_probs is not None else 0.0

        del X_train_tensor, y_train_tensor
        del X_val_tensor, y_val_tensor
        del X_test_tensor, y_test_tensor
        del train_loader, val_loader, test_loader
        del optimizer, fold_model, all_probs, y_probs, y_pred
        if best_state_dict is not None:
            del best_state_dict

        free_memory()

    final_metrics = aggregate_classification_cv_metrics(
        accuracy=float(np.nanmean(accs)),
        precision=float(np.nanmean(precs)),
        recall=float(np.nanmean(recalls)),
        f1_score_value=float(np.nanmean(f1s)),
        roc_auc=float(np.nanmean(aucs)),
        training_time=float(np.nansum(fit_times)),
        name=model_name,
        y_true=y,
        y_pred=oof_pred,
        y_probs=oof_probs if has_probs else None,
    )

    final_metrics.estimators = estimators
    p.plot_classification_results(final_metrics, model_name="PyTorch CV")

    return final_metrics


class TransformerTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_collate_fn(tokenizer, max_length: int = 256):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encodings["labels"] = labels
        return encodings

    return collate_fn


def cross_validate_transformer_model(
    model: nn.Module,
    tokenizer,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv,
    optimizer_class: type,
    optimizer_params: dict,
    num_epochs: int = 5,
    batch_size: int = 16,
    max_length: int = 256,
    device: str | None = None,
    enable_plot: bool = False,
    model_name: str = "Transformer CV",
    early_stopping_patience: int = 2,
    early_stopping_min_delta: float = 1e-4,
    validation_size: float = 0.1,
    restore_best_weights: bool = True,
):
    device = torch.device(device or get_device())

    fit_times, accs, f1s, precs, recalls, aucs = [], [], [], [], [], []
    estimators = []

    X = np.array(X)
    y = np.array(y)

    n_samples = len(y)
    classes = np.unique(y)
    n_classes = len(classes)

    oof_pred = np.empty(n_samples, dtype=int)
    oof_probs = np.empty((n_samples, n_classes), dtype=float)

    collate_fn = make_collate_fn(tokenizer, max_length=max_length)

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(f"\nFold {fold_index + 1}/{cv.get_n_splits()}")

        X_train_full, y_train_full = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=validation_size,
            random_state=42,
            stratify=y_train_full,
        )

        train_dataset = TransformerTextDataset(X_train, y_train)
        val_dataset = TransformerTextDataset(X_val, y_val)
        test_dataset = TransformerTextDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        fold_model = copy.deepcopy(model).to(device)
        optimizer = optimizer_class(fold_model.parameters(), **optimizer_params)

        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = 0
        epochs_without_improvement = 0

        start_fit = perf_counter()

        for epoch in tqdm(
            range(num_epochs), desc=f"Training fold {fold_index + 1}", leave=True
        ):
            fold_model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                optimizer.zero_grad()

                outputs = fold_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            fold_model.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        device, non_blocking=True
                    )
                    labels = batch["labels"].to(device, non_blocking=True)

                    outputs = fold_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                    total_val_loss += outputs.loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                if restore_best_weights:
                    best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in fold_model.state_dict().items()
                    }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                tqdm.write(
                    f"Early stopping on epoch {epoch + 1} "
                    f"(best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})"
                )
                break

        if restore_best_weights and best_state_dict is not None:
            fold_model.load_state_dict(best_state_dict)
            fold_model.to(device)

        fit_times.append(perf_counter() - start_fit)

        fold_model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                outputs = fold_model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(outputs.logits, dim=1).cpu()
                all_probs.append(probs)

        y_probs = torch.cat(all_probs, dim=0).numpy()
        y_pred = np.argmax(y_probs, axis=1)

        fold_metrics = evaluate_classification(
            y_true=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            model_name=f"fold_{fold_index + 1}",
            enable_plot=enable_plot,
        )

        accs.append(fold_metrics.accuracy)
        f1s.append(fold_metrics.f1_score)
        precs.append(fold_metrics.precision)
        recalls.append(fold_metrics.recall)
        aucs.append(
            fold_metrics.roc_auc if fold_metrics.roc_auc is not None else float("nan")
        )

        estimators.append(
            {
                "tokenizer_name": tokenizer.name_or_path,
                "model": copy.deepcopy(fold_model).cpu(),
            }
        )

        oof_pred[test_index] = y_pred
        oof_probs[test_index] = y_probs

        del train_loader, val_loader, test_loader, optimizer, fold_model
        if best_state_dict is not None:
            del best_state_dict

        free_memory()

    final_metrics = aggregate_classification_cv_metrics(
        accuracy=float(np.nanmean(accs)),
        precision=float(np.nanmean(precs)),
        recall=float(np.nanmean(recalls)),
        f1_score_value=float(np.nanmean(f1s)),
        roc_auc=float(np.nanmean(aucs)),
        training_time=float(np.nansum(fit_times)),
        name=model_name,
        y_true=y,
        y_pred=oof_pred,
        y_probs=oof_probs,
    )

    final_metrics.estimators = estimators
    p.plot_classification_results(final_metrics, model_name=model_name)

    return final_metrics


def fine_tune_and_validate(
    model: nn.Module,
    X,
    y,
    *,
    criterion: nn.Module,
    optimizer_class: type,
    optimizer_params: dict,
    preprocessor=None,
    num_epochs: int = 5,
    batch_size: int = 32,
    device: str | None = None,
    enable_plot: bool = True,
    model_name: str = "PyTorch Model",
):

    model = model.to(device)

    X = preprocessor.transform(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.long).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True
    )

    optimizer = optimizer_class(model.parameters(), **optimizer_params)

    fit_times = []
    model.train()

    start_fit = perf_counter()
    for epoch in tqdm(range(num_epochs), desc="Training", leave=True):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"  Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")
    fit_times.append(perf_counter() - start_fit)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        y_score = torch.sigmoid(logits).cpu().numpy().flatten()

    y_pred = _labels_from_score(y_score, 0.5)
    y_probs = _positive_class_probabilities(y_score)

    metrics = evaluate_classification(
        y_true=y,
        y_pred=y_pred,
        y_probs=y_probs,
        model_name=model_name,
        enable_plot=enable_plot,
    )

    final_metrics = aggregate_classification_cv_metrics(
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score_value=metrics.f1_score,
        roc_auc=metrics.roc_auc if metrics.roc_auc is not None else float("nan"),
        training_time=float(np.nansum(fit_times)),
        name=model_name,
        y_true=y,
        y_pred=y_pred,
        y_probs=y_probs,
    )

    p.plot_classification_results(final_metrics, model_name="Fine-tuned model")

    return final_metrics


@torch.no_grad()
def predict_with_ensemble(ensemble, X, device, threshold=0.5, batch_size=128):
    all_probs = []

    for start in tqdm(range(0, len(X), batch_size), desc="Predicting ensemble"):
        end = start + batch_size
        X_batch_raw = X[start:end]

        fold_probs = []

        for estimator in ensemble:
            preprocessor = estimator["preprocessor"]
            model = estimator["model"]

            X_batch = preprocessor.transform(X_batch_raw)

            X_batch = torch.tensor(X_batch, dtype=torch.long, device=device)

            model.eval()
            probs = torch.sigmoid(model(X_batch)).detach().cpu().numpy().flatten()
            fold_probs.append(probs)

        mean_probs = np.mean(fold_probs, axis=0)
        all_probs.append(mean_probs)

    all_probs = np.concatenate(all_probs)
    predictions = (all_probs >= threshold).astype(int)
    return predictions, all_probs
