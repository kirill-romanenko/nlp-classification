from typing import Optional, Any, Dict

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from core.baseline_training import extract_feature_names_from_pipeline
from core.training_results import ClassificationMetrics, BaseMetrics

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_PALETTE = "viridis"
DEFAULT_GRID_ALPHA = 0.3


def _setup_plot_style(figsize: tuple[int, int] = DEFAULT_FIGSIZE) -> None:
    plt.figure(figsize=figsize)
    plt.grid(alpha=DEFAULT_GRID_ALPHA)


def _finalize_plot(title: str, x_label: str = "", y_label: str = "") -> None:
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    if x_label:
        plt.xlabel(x_label, fontsize=12)
    if y_label:
        plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_histogram_numeric(data: pd.DataFrame, feature: str,
                           figsize: tuple[int, int] = (8, 4),
                           x_min: Optional[float] = None, x_max: Optional[float] = None) -> None:
    filtered = data.copy()
    if x_min:
        filtered = filtered[filtered[feature] >= x_min]
    if x_max:
        filtered = filtered[filtered[feature] <= x_max]

    _setup_plot_style(figsize=figsize)
    sns.histplot(filtered[feature], kde=True)
    _finalize_plot(f'Distribution of {feature}', feature, 'Frequency')


def barplot(
        category_counts: pd.Series,
        title: str,
        y_label: str,
        figsize: tuple[int, int] = (4, 6),
        top_n: Optional[int] = None,
        color_palette: str = DEFAULT_PALETTE
) -> None:
    if top_n is not None and len(category_counts) > top_n:
        plot_data = category_counts.nlargest(top_n)
    else:
        plot_data = category_counts

    plt.figure(figsize=figsize)
    plt.grid(axis='x', alpha=DEFAULT_GRID_ALPHA)
    sns.barplot(x=plot_data.values,
                y=plot_data.index,
                hue=plot_data.index,
                palette=color_palette,
                orient='h',
                legend=False,
                dodge=False)
    _finalize_plot(title, 'Frequency', y_label)


def plot_wordcloud(
    token_counts: pd.Series,
    title: str = "Word Cloud",
    figsize: tuple[int, int] = (12, 8),
    max_words: int = 100,
    background_color: str = "white",
    colormap: str = "viridis",
    width: int = 800,
    height: int = 400
) -> None:
    word_frequencies = token_counts.to_dict()
    wordcloud = WordCloud(
        background_color=background_color,
        width=width,
        height=height,
        max_words=max_words,
        colormap=colormap,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(word_frequencies)

    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_pie_chart(data: pd.DataFrame, column: str, title: str) -> None:
    counts = data[column].value_counts(dropna=False)
    colors = sns.color_palette(DEFAULT_PALETTE, len(counts))
    plt.figure(figsize=(7, 7))
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    plt.setp(autotexts, size=12, weight="bold", color="black")
    plt.setp(texts, size=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.show()


def plot_feature_importance(model: BaseEstimator, model_name: str, feature_names: list[str],
                            top_n: Optional[int] = None,
                            figsize: tuple[int, int] = (10, 6),
                            model_type: str = 'auto') -> pd.DataFrame:
    if model_type == 'auto':
        if hasattr(model, 'feature_importances_'):
            model_type = 'tree'
        elif hasattr(model, 'coef_'):
            model_type = 'linear'
        else:
            raise ValueError("Could not determine model type automatically. Please specify 'tree' or 'linear'")

    match model_type:
        case 'tree':
            importances = model.feature_importances_
            importance_label = "Feature Importance"
        case 'linear':
            if len(model.coef_.shape) > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_[0])
            importance_label = "Absolute Coefficient"
        case _:
            raise ValueError("model_type must be either 'tree' or 'linear'")

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    if top_n:
        feature_importances = feature_importances.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(x='Importance', y='Feature',
                data=feature_importances, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Feature Importances for {model_name}')
    plt.xlabel(importance_label)
    plt.tight_layout()
    plt.show()

    return feature_importances

def plot_feature_importance_cv(pipeline: Any, model_name: str,
                               feature_names: Optional[list[str]], X: Any, y: Any) -> None:
    try:
        if isinstance(pipeline, Pipeline):
            final_model = pipeline.named_steps['model']
        else:
            final_model = pipeline

        pipeline.fit(X, y)

        if hasattr(final_model, 'feature_importances_') or hasattr(final_model, 'coef_'):
            if hasattr(final_model, 'feature_importances_'):
                n_features = len(final_model.feature_importances_)
            else:
                n_features = final_model.coef_.shape[1] \
                    if len(final_model.coef_.shape) > 1 else final_model.coef_.shape[0]
            actual_feature_names = extract_feature_names_from_pipeline(pipeline, X, n_features, feature_names)
            plot_feature_importance(final_model, model_name, actual_feature_names, top_n=20)
        else:
            print(f"Warning: Model {model_name} does not support feature importance")
    except Exception as e:
        print(f"Warning: Could not plot feature importance for {model_name}: {str(e)}")


def plot_classification_results(metrics: ClassificationMetrics, model_name: str = "Model") -> None:
    plt.figure(figsize=(15, 6))

    # if metrics.confusion_matrix is not None:
    #     plt.subplot(1, 2, 1)
    #     sns.heatmap(metrics.confusion_matrix, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=['Predicted Negative', 'Predicted Positive'],
    #                 yticklabels=['Actual Negative', 'Actual Positive'])
    #     plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
    #     plt.xlabel('Predicted Label', fontsize=12)
    #     plt.ylabel('True Label', fontsize=12)

    if metrics.confusion_matrix is not None:
        plt.subplot(1, 2, 1)
        cm = metrics.confusion_matrix
        n_classes = cm.shape[0]
        # Динамические подписи
        if n_classes == 2:
            xticklabels = ['Predicted Negative', 'Predicted Positive']
            yticklabels = ['Actual Negative', 'Actual Positive']
        else:
            xticklabels = [f'Predicted {i}' for i in range(n_classes)]
            yticklabels = [f'Actual {i}' for i in range(n_classes)]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=xticklabels,
                    yticklabels=yticklabels)
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

    if metrics.roc_curve is not None and metrics.roc_auc is not None:
        plt.subplot(1, 2, 2)
        plt.plot(metrics.roc_curve.fpr, metrics.roc_curve.tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {metrics.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(metrics: list[BaseMetrics],
                         title: str = 'Model Evaluation Metrics Comparison',
                         figsize: tuple[int, int] = (12, 6)) -> None:
    rows = {}
    for i, metric in enumerate(metrics):
        row_name = metric.name if getattr(metric, 'name', None) else f'Model {i + 1}'
        rows[row_name] = metric.to_compact_dict()

    metrics_df = pd.DataFrame.from_dict(rows, orient='index')

    plt.figure(figsize=figsize)

    normalized_df = metrics_df.copy()
    for column in metrics_df.columns:
        min_value = metrics_df[column].min()
        max_value = metrics_df[column].max()
        if min_value != max_value:
            normalized_df[column] = (metrics_df[column] - min_value) / (max_value - min_value)
            continue
        normalized_df[column] = 0.5

    sns.heatmap(normalized_df, annot=metrics_df, fmt='.3f', cmap='RdBu_r',
                cbar_kws={'label': 'Normalized Score (0-1 per metric)'})
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compare_metrics_heatmap(df1: pd.DataFrame, df2: pd.DataFrame, df1_name: str = 'DF1', df2_name: str = 'DF2',
                            figsize: tuple[int, int] = (12, 6), annot_fontsize: int = 10,
                            title: str = "Comparison of ML Metrics",
                            lower_is_better_metrics: Optional[list[str]] = None) -> tuple[Any, pd.DataFrame]:
    if lower_is_better_metrics is None:
        lower_is_better_patterns = [
            'time', 'loss', 'error', 'cost', 'latency', 'duration',
            'mse', 'mae', 'rmse', 'runtime', 'seconds', 'minutes'
        ]
        lower_is_better_metrics = []
        for column in df1.columns:
            column_lower = column.lower()
            if any(pattern in column_lower for pattern in lower_is_better_patterns):
                lower_is_better_metrics.append(column)

    delta = df2 - df1

    time_patterns = ['time', 'duration', 'runtime', 'seconds', 'minutes']
    time_columns = []

    for column in df1.columns:
        column_lower = column.lower()
        if any(pattern in column_lower for pattern in time_patterns):
            time_columns.append(column)
            pct_column_name = f"{column} Change (%)"
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_change = np.where(df1[column] != 0, (delta[column] / df1[column]) * 100, 0)
            delta[pct_column_name] = pct_change

            if column in lower_is_better_metrics:
                lower_is_better_metrics.append(pct_column_name)

    semantic_delta = delta.copy()
    for column in lower_is_better_metrics:
        if column in semantic_delta.columns:
            semantic_delta[column] = -semantic_delta[column]

    normalized_delta = semantic_delta.copy()
    for column in semantic_delta.columns:
        column_values = semantic_delta[column]
        min_value = column_values.min()
        max_value = column_values.max()

        if min_value != max_value:
            normalized_column = column_values.copy()

            positive_mask = column_values > 0
            if max_value > 0 and positive_mask.any():
                normalized_column[positive_mask] = column_values[positive_mask] / max_value

            negative_mask = column_values < 0
            if max_value < 0 and negative_mask.any():
                normalized_column[negative_mask] = column_values[negative_mask] / abs(min_value)

            normalized_delta[column] = normalized_column
            continue
        normalized_delta[column] = 0

    colors = ["#ff2700", "#ffffff", "#00b975"]
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(normalized_delta, annot=delta, fmt='.3f', cmap=cmap, center=0, linewidths=.5,
                ax=ax, annot_kws={"size": annot_fontsize},
                cbar_kws={'label': 'Improvement (Green) ← → Degradation (Red)'})

    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

    return delta


def print_classification_report(metrics: Dict[str, Any], model_name: str = 'Model'):
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'],
        'Value': [
            f'{metrics["ROC AUC"]:.4f}' if metrics["ROC AUC"] is not None else 'N/A',
            f'{metrics["F1 Score"]:.4f}',
            f'{metrics["Precision"]:.4f}',
            f'{metrics["Recall"]:.4f}',
            f'{metrics["Accuracy"]:.4f}'
        ]
    })

    classification_report_df = pd.DataFrame(metrics['Classification Report'])

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} EVALUATION".center(60))
    print("=" * 60)

    print("\nMAIN METRICS:")
    print(metrics_df.to_string(index=False))

    print("\n\nCLASSIFICATION REPORT:")
    print(classification_report_df.to_string(index=False))

    print("\n" + "=" * 60)
