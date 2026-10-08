"""Сборка, оценка и журналирование sklearn-экспериментов Titanic."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# Повторный экспорт сохраняет прежние импорты из ноутбуков.
from src.preprocessing import (
    ExperimentConfig,
    ProcessingType,
    base_config,
    build_preprocessor,
    prepare_features,
)


def evaluate_model(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Собирает pipeline из df и config, возвращает средние CV-метрики и std.

    target — имя колонки меток. experiment_name включает запись в CSV-журнал.
    Предобработка обучается внутри каждого train-фолда.
    """
    if config is None:
        config = base_config
    X, y, pipeline = build_experiment(df, target, model, config)
    return evaluate_cv(pipeline, X, y, model, config, experiment_name)


def evaluate_cv(
    estimator,
    X,
    y,
    model,
    config,
    experiment_name=None,
    fit_params=None,
):
    """Оценивает estimator на пяти фолдах; возвращает Accuracy/ROC-AUC mean/std.

    X и y — признаки и метки в одинаковом порядке; fit_params передаются в fit.
    model и config описывают запись журнала, если задан experiment_name.
    """
    result = cross_validate(
        estimator, X, y, cv=get_cv(), scoring=["accuracy", "roc_auc"],
        params=fit_params,
    )
    metrics = {
        "accuracy_mean": result["test_accuracy"].mean(),
        "accuracy_std": result["test_accuracy"].std(),
        "roc_auc_mean": result["test_roc_auc"].mean(),
        "roc_auc_std": result["test_roc_auc"].std(),
    }
    if experiment_name is not None:
        save_experiment(experiment_name, model, config, metrics)
    return metrics


def save_experiment(
    name: str,
    model: BaseEstimator | str,
    config: dict,
    metrics: dict,
    path: str = "experiments/experiments.csv",
    model_params: dict | None = None,
) -> None:
    """Добавляет одну строку с именем, параметрами, обработкой и метриками в CSV.

    Для model в виде строки параметры нужно передать через model_params.
    Существующие строки сохраняются; повторное имя добавляет новую запись.
    """
    if isinstance(model, str):
        if model_params is None:
            raise ValueError("model_params is required when model is a name")
        model_name = model
        params = model_params
    else:
        model_name = model.__class__.__name__
        params = model.get_params() if model_params is None else model_params

    experiment = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "name": name,
        "model": model_name,
        "model_params": json.dumps(params, default=str),
        "config": json.dumps(config, default=str),
        **metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([experiment]).to_csv(
        path, mode="a", header=not path.exists(), index=False,
    )


def get_cv() -> StratifiedKFold:
    """Возвращает общие пять стратифицированных фолдов: shuffle=True, seed=42."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_experiment(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Возвращает признаки X, метки y и необученный pipeline обработки/model.

    df содержит target и исходные признаки Titanic; config задаёт их обработку.
    """
    if config is None:
        config = base_config
    X, numeric, binary, onehot = prepare_features(df, config)
    y = df[target]
    preprocessor = build_preprocessor(numeric, binary, onehot, config)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return X, y, pipeline


def tune_model(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    param_grid: dict,
    config: ExperimentConfig | None = None,
) -> GridSearchCV:
    """Перебирает param_grid на общей CV и возвращает обученный GridSearchCV.

    Параметры модели задаются с префиксом model__; выбор и refit — по Accuracy.
    Лучший pipeline переобучается на всём df, а не на отдельном фолде.
    """
    X, y, pipeline = build_experiment(df, target, model, config)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
        cv=get_cv(),
        n_jobs=-1,
        refit="accuracy",
        verbose=0,
    )
    grid_search.fit(X, y)
    return grid_search


def save_cv_experiment(
    name: str,
    cv_results: pd.DataFrame,
    model_name: str,
    params: dict,
    config: dict,
    metric_columns: dict[str, str],
    selection_metric: str = "accuracy_mean",
    higher_is_better: bool = True,
    iteration_column: str | None = None,
    iteration_offset: int = 0,
    iteration_param: str | None = None,
    cv_method: str = "cv",
) -> dict:
    """Выбирает строку таблицы CV, сохраняет её в журнал и возвращает метрики.

    metric_columns связывает имена метрик с колонками cv_results.
    iteration_column и offset переводят выбранную итерацию в число деревьев;
    iteration_param задаёт имя параметра модели для записи этого числа.
    """
    if selection_metric not in metric_columns:
        raise ValueError("selection_metric must be present in metric_columns")
    score_column = metric_columns[selection_metric]
    if higher_is_better:
        best_idx = cv_results[score_column].idxmax()
    else:
        best_idx = cv_results[score_column].idxmin()
    row = cv_results.loc[best_idx]
    metrics = {metric: row[column] for metric, column in metric_columns.items()}
    experiment_config = {
        **config, "cv_method": cv_method, "selection_metric": selection_metric,
    }
    saved_params = params.copy()
    if iteration_column is not None:
        selected_iterations = int(row[iteration_column]) + iteration_offset
        experiment_config["selected_iterations"] = selected_iterations
        if iteration_param is not None:
            experiment_config["cv_max_iterations"] = params.get(iteration_param)
            saved_params[iteration_param] = selected_iterations
    save_experiment(
        name, model_name, experiment_config, metrics, model_params=saved_params,
    )
    return metrics
