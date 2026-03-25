import time

import numpy as np

import core.visualization as p

from typing import Any, Optional, Union, Callable

from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from core.training_results import MultipleModelsResults, BaseMetrics, ClassificationMetrics, RocCurveData, \
    RegressionMetrics, GenericMetrics, ClassificationReportRow


def aggregate_regression_cv_metrics(
    *,
    mae: Optional[float] = None,
    mse: Optional[float] = None,
    rmse: Optional[float] = None,
    r2: Optional[float] = None,
    explained_variance: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
) -> RegressionMetrics:
    computed_rmse = rmse
    if computed_rmse is None and mse is not None:
        try:
            computed_rmse = float(mse ** 0.5)
        except Exception:
            computed_rmse = None

    metrics = RegressionMetrics(
        mae=float(mae) if mae is not None else float('nan'),
        mse=float(mse) if mse is not None else float('nan'),
        rmse=float(computed_rmse) if computed_rmse is not None else float('nan'),
        r2=float(r2) if r2 is not None else float('nan'),
        explained_variance=float(explained_variance) if explained_variance is not None else float('nan'),
        training_time=float(training_time) if training_time is not None else None,
        name=name,
    )

    return metrics


def aggregate_classification_cv_metrics(
    *,
    accuracy: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1_score_value: Optional[float] = None,
    roc_auc: Optional[float] = None,
    training_time: Optional[float] = None,
    name: Optional[str] = None,
    y_true: Optional[Union[np.ndarray, Any]] = None,
    y_pred: Optional[Union[np.ndarray, Any]] = None,
    y_probs: Optional[Union[np.ndarray, Any]] = None,
) -> ClassificationMetrics:
    # cm = None
    # roc_curve_data = None

    # if y_true is not None and y_pred is not None:
    #     cm = confusion_matrix(y_true, y_pred)

    # if y_true is not None and y_probs is not None:
    #     fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    #     roc_curve_data = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
    #     if roc_auc is None:
    #         try:
    #             roc_auc = float(roc_auc_score(y_true, y_probs))
    #         except Exception:
    #             roc_auc = None

    # metrics = ClassificationMetrics(
    #     roc_auc=float(roc_auc) if roc_auc is not None else None,
    #     f1_score=float(f1_score_value) if f1_score_value is not None else float('nan'),
    #     precision=float(precision) if precision is not None else float('nan'),
    #     recall=float(recall) if recall is not None else float('nan'),
    #     accuracy=float(accuracy) if accuracy is not None else float('nan'),
    #     confusion_matrix=cm,
    #     training_time=float(training_time) if training_time is not None else None,
    #     name=name,
    # )

    # if roc_curve_data is not None:
    #     metrics.roc_curve = roc_curve_data

    # return metrics

    cm = None
    roc_curve_data = None
    final_roc_auc = roc_auc  # используем переданное, если есть

    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)

    # Определяем количество классов
    n_classes = None
    if y_probs is not None and hasattr(y_probs, 'ndim') and y_probs.ndim == 2:
        n_classes = y_probs.shape[1]
    elif y_true is not None:
        n_classes = len(np.unique(y_true))
    else:
        n_classes = 2  # значение по умолчанию

    # Вычисляем ROC AUC, если не передан явно
    if y_true is not None and y_probs is not None and final_roc_auc is None:
        try:
            if n_classes == 2:
                # Бинарный случай: извлекаем вероятности положительного класса
                if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                    probs_pos = y_probs[:, 1]
                elif y_probs.ndim == 1:
                    probs_pos = y_probs
                else:
                    probs_pos = y_probs  # fallback

                final_roc_auc = float(roc_auc_score(y_true, probs_pos))
                # Также вычисляем ROC-кривую для бинарного случая
                fpr, tpr, thresholds = roc_curve(y_true, probs_pos)
                roc_curve_data = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)
            else:
                # Многоклассовый случай: macro-average over one-vs-rest
                final_roc_auc = float(roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro'))
                # Для многоклассового случая не сохраняем единую ROC-кривую
                # (можно расширить структуру для списка кривых, но для базового функционала оставляем)
        except Exception as e:
            # Не удалось вычислить AUC (например, недостаточно классов или другие проблемы)
            final_roc_auc = None

    metrics = ClassificationMetrics(
        roc_auc=final_roc_auc,
        f1_score=float(f1_score_value) if f1_score_value is not None else float('nan'),
        precision=float(precision) if precision is not None else float('nan'),
        recall=float(recall) if recall is not None else float('nan'),
        accuracy=float(accuracy) if accuracy is not None else float('nan'),
        confusion_matrix=cm,
        training_time=float(training_time) if training_time is not None else None,
        name=name,
    )

    if roc_curve_data is not None:
        metrics.roc_curve = roc_curve_data

    return metrics


def _evaluate_multiple_models_pydantic(models: list[tuple[str, BaseEstimator]],
                                       evaluation_func: Callable, task_type: str,
                                       *args, **kwargs) -> MultipleModelsResults:
    metrics_objects = []
    for model_name, model in models:
        current_model = clone(model)
        eval_result = evaluation_func(current_model, model_name, *args, **kwargs)

        if isinstance(eval_result, BaseMetrics):
            eval_result.name = model_name
            metrics_objects.append(eval_result)
            continue

        generic_metrics = GenericMetrics(values=eval_result, name=model_name)
        metrics_objects.append(generic_metrics)

    p.plot_metrics_heatmap(metrics_objects)

    return MultipleModelsResults(results=metrics_objects, task_type=task_type)


def train_evaluate_model_cv(model: BaseEstimator, model_name: str, X: Any, y: Any,
                            preprocessor: Optional[Any] = None, cv: int = 5,
                            seed: Optional[int] = None, feature_names: Optional[list[str]] = None,
                            plot_feature_importance: bool = True,
                            task_type: str = "classification") -> BaseMetrics:
    if isinstance(preprocessor, Pipeline):
        steps = preprocessor.steps.copy()
        steps.append(("model", model))
        pipline = Pipeline(steps)
    elif preprocessor is not None:
        pipline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    else:
        pipline = model

    match task_type:
        case "classification":
            scoring = {
                "accuracy": "accuracy",
                "precision": make_scorer(precision_score, average="macro", zero_division=0),
                "recall": make_scorer(recall_score, average="macro", zero_division=0),
                "f1": make_scorer(f1_score, average="macro", zero_division=0),
                "roc_auc": make_scorer(roc_auc_score, average="macro", multi_class="ovr", response_method="predict_proba")
            }
        case "regression":
            scoring = {
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
                "r2": "r2",
                "explained_variance": "explained_variance"
            }
        case _:
            raise ValueError(f"Unsupported task_type: {task_type}. Must be 'classification' or 'regression'")

    start_time = time.time()
    cv_results = cross_validate(
        pipline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )
    end_time = time.time()
    training_time = end_time - start_time

    match task_type:
        case "classification":
            cv_metrics = aggregate_classification_cv_metrics(
                accuracy=float(cv_results['test_accuracy'].mean()),
                precision=float(cv_results['test_precision'].mean()),
                recall=float(cv_results['test_recall'].mean()),
                f1_score_value=float(cv_results['test_f1'].mean()),
                roc_auc=float(cv_results['test_roc_auc'].mean()),
                training_time=float(training_time),
                name=model_name
            )
            p.plot_classification_results(cv_metrics, model_name)

        case "regression":
            cv_metrics = aggregate_regression_cv_metrics(
                mae=float(-cv_results['test_mae'].mean()),
                mse=float(-cv_results['test_mse'].mean()),
                r2=float(cv_results['test_r2'].mean()),
                explained_variance=float(cv_results['test_explained_variance'].mean()),
                training_time=float(training_time),
                name=model_name
            )

    if plot_feature_importance:
        p.plot_feature_importance_cv(pipline, model_name, feature_names, X, y)

    return cv_metrics, pipline


def train_evaluate_models_cv(models: list[tuple[str, BaseEstimator]], X: Any, y: Any,
                             preprocessor: Optional[Any] = None, cv: int = 5,
                             seed: Optional[int] = None, feature_names: Optional[list[str]] = None,
                             plot_feature_importance: bool = True,
                             task_type: str = "classification") -> MultipleModelsResults:
    def _cv_evaluation_wrapper(model: BaseEstimator, model_name: str) -> BaseMetrics:
        current_preprocessor = clone(preprocessor) if preprocessor else None
        metrics, _ = train_evaluate_model_cv(
            model, model_name, X, y, current_preprocessor, cv, seed,
            feature_names, plot_feature_importance, task_type
        )
        return metrics

    return _evaluate_multiple_models_pydantic(models, _cv_evaluation_wrapper, task_type)


def extract_feature_names_from_pipeline(
        pipeline: Any,
        X: Any,
        n_features: int,
        provided_feature_names: Optional[list[str]] = None
) -> list[str]:
    # If provided feature names match the number of features, use them
    if provided_feature_names is not None and len(provided_feature_names) == n_features:
        return provided_feature_names

    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) == 1:
        if hasattr(X, 'columns') and len(X.columns) == n_features:
            return list(X.columns)
        else:
            return [f'feature_{i}' for i in range(n_features)]

    feature_names = []

    preprocessor_steps = pipeline.steps[:-1]  # All steps except the last (model)

    if len(preprocessor_steps) == 1:
        step_name, preprocessor = preprocessor_steps[0]
        feature_names = extract_feature_names_from_transformer(preprocessor, step_name)
    else:
        for step_name, transformer in preprocessor_steps:
            step_feature_names = extract_feature_names_from_transformer(transformer, step_name)
            feature_names.extend(step_feature_names)

    if len(feature_names) != n_features:
        if hasattr(X, 'columns') and len(X.columns) == n_features:
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(n_features)]

    return feature_names


def extract_feature_names_from_transformer(transformer: Any, step_name: str) -> list[str]:
    try:
        if hasattr(transformer, 'transformers_'):
            feature_names = []
            for name, trans, columns in transformer.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    trans_feature_names = trans.get_feature_names_out()
                    if isinstance(columns, str):
                        prefix = f"{name}_{columns}_"
                    else:
                        prefix = f"{name}_"
                    prefixed_names = [f"{prefix}{name}" for name in trans_feature_names]
                    feature_names.extend(prefixed_names)
            return feature_names

        elif hasattr(transformer, 'get_feature_names_out'):
            return list(transformer.get_feature_names_out())

        elif hasattr(transformer, 'vocabulary_'):
            return list(transformer.vocabulary_.keys())

        elif hasattr(transformer, 'feature_names_in_'):
            return list(transformer.feature_names_in_)

        return []

    except Exception:
        return []


def _calculate_classification_metrics(y_true: Any, y_pred: Any, y_probs: Optional[Any] = None) -> ClassificationMetrics:
    # roc_auc_value = roc_auc_score(y_true, y_probs) if y_probs is not None else None
    roc_auc_value = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    metrics = ClassificationMetrics(
        roc_auc=roc_auc_value,
        f1_score=f1_score(y_true, y_pred, average='macro', zero_division=0),
        precision=precision_score(y_true, y_pred, average='macro'),
        recall=recall_score(y_true, y_pred, average='macro'),
        accuracy=(y_pred == y_true).mean(),
        confusion_matrix=confusion_matrix(y_true, y_pred)
    )

    # report_rows = [
    #     ClassificationReportRow(
    #         class_label='Positive',
    #         precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    #         recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    #     ),
    #     ClassificationReportRow(
    #         class_label='Negative',
    #         precision=precision_score(y_true, y_pred, pos_label=0, zero_division=0),
    #         recall=recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    #     )
    # ]

    # metrics.classification_report = report_rows

    # if y_probs is not None:
    #     fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    #     metrics.roc_curve = RocCurveData(fpr=fpr, tpr=tpr, thresholds=thresholds)

    # Формируем строки отчёта для каждого класса
    report_rows = []
    unique_classes = sorted(np.unique(y_true))
    for cls in unique_classes:
        report_rows.append(
            ClassificationReportRow(
                class_label=str(cls),
                precision=precision_score(y_true, y_pred, labels=[cls], average='micro', zero_division=0),
                recall=recall_score(y_true, y_pred, labels=[cls], average='micro', zero_division=0)
            )
        )

    metrics.classification_report = report_rows

    return metrics


def evaluate_classification(y_true: Any, y_pred: Any, y_probs: Optional[Any] = None, model_name: str = "Model",
                            enable_plot: bool = True) -> ClassificationMetrics:
    metrics = _calculate_classification_metrics(y_true, y_pred, y_probs)
    if enable_plot:
        p.plot_classification_results(metrics, model_name)
        p.print_classification_report(metrics.to_report_dict(), model_name)

    return metrics
