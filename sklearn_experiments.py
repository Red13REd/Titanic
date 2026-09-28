import pandas as pd
import numpy as np

import json
from pathlib import Path
from datetime import datetime

from typing import TypedDict, Literal, NotRequired

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



ProcessingType = Literal[
    "numeric",
    "binary",
    "onehot",
    "_",
]


class ExperimentConfig(TypedDict):
    features: dict[str, ProcessingType]

    family_size: NotRequired[bool]
    is_alone: NotRequired[bool]
    is_child: NotRequired[bool]
    fare_log1p: NotRequired[bool]
    age_squared: NotRequired[bool]
    sex_pclass: NotRequired[bool]
    scale_numeric: NotRequired[bool]

base_config: ExperimentConfig = {
    "features": {
        "Age": "numeric",
        "Fare": "numeric",
        "Sex": "binary",
        "Embarked": "onehot",
        "Pclass": "numeric",
        "SibSp": "numeric",
        "Parch": "numeric",
    },
    "scale_numeric": True,
}


def prepare_features(
    df: pd.DataFrame,
    config: ExperimentConfig | None = None,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    if config is None:
        config = base_config

    numeric_features: list[str] = []
    binary_features: list[str] = []
    onehot_features: list[str] = []

    # Выбираем только активные признаки.
    active_features = {
        feature: processing_type
        for feature, processing_type in config["features"].items()
        if not feature.startswith("_")
        and processing_type != "_"
    }

    selected_features = list(active_features.keys())
    required_features = selected_features.copy()

    # Зависимости для feature engineering.
    # Эти колонки нужны для вычислений, даже если отключены в features.
    dependencies = {
        "family_size": ["SibSp", "Parch"],
        "is_alone": ["SibSp", "Parch"],
        "is_child": ["Age"],
        "age_squared": ["Age"],
        "fare_log1p": ["Fare"],
        "sex_pclass": ["Sex", "Pclass"],
    }

    for flag, columns in dependencies.items():
        if config.get(flag, False):

            for column in columns:
                if column not in required_features:
                    required_features.append(column)

    X = df[required_features].copy()

    # Распределяем признаки по типам обработки.
    for feature, processing_type in active_features.items():

        if processing_type == "numeric":
            numeric_features.append(feature)

        elif processing_type == "binary":
            binary_features.append(feature)

        elif processing_type == "onehot":
            onehot_features.append(feature)

        else:
            raise ValueError(
                f"Unknown processing type: {processing_type}"
            )

    # Feature engineering

    if config.get("family_size", False):
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        numeric_features.append("FamilySize")
        selected_features.append("FamilySize")

    if config.get("is_alone", False):
        X["IsAlone"] = (
            X["SibSp"] + X["Parch"] == 0
        ).astype(int)

        numeric_features.append("IsAlone")
        selected_features.append("IsAlone")

    if config.get("is_child", False):
        X["IsChild"] = (
            (X["Age"] < 16)
            .astype(float)
            .where(X["Age"].notna())
        )

        numeric_features.append("IsChild")
        selected_features.append("IsChild")

    if config.get("fare_log1p", False):
        X["FareLog1p"] = np.log1p(X["Fare"])

        numeric_features.append("FareLog1p")
        selected_features.append("FareLog1p")

    if config.get("age_squared", False):
        X["AgeSquared"] = X["Age"] ** 2

        numeric_features.append("AgeSquared")
        selected_features.append("AgeSquared")

    if config.get("sex_pclass", False):
        X["SexPclass"] = (
            X["Sex"].astype(str)
            + "_"
            + X["Pclass"].astype(str)
        )

        onehot_features.append("SexPclass")
        selected_features.append("SexPclass")

    # Оставляем только те колонки,
    # которые действительно пойдут в модель.
    X = X[selected_features]

    return (
        X,
        numeric_features,
        binary_features,
        onehot_features,
    )

def build_preprocessor(
    numeric_features: list[str],
    binary_features: list[str],
    onehot_features: list[str],
    config: ExperimentConfig | None = None,
) -> ColumnTransformer:

    if config is None:
        config = base_config

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if config.get("scale_numeric", True):
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "binary",
            OneHotEncoder(
                drop="if_binary",
                handle_unknown="ignore",
            ),
            binary_features,
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
            onehot_features,
        ),
    ])



def evaluate_model(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    config: ExperimentConfig | None = None,
    experiment_name: str | None = None,
) -> dict:

    if config is None:
        config = base_config

    X, y, pipeline = build_experiment(
        df=df,
        target=target,
        model=model,
        config=config,
    )

    return evaluate_cv(
        estimator=pipeline,
        X=X,
        y=y,
        model=model,
        config=config,
        experiment_name=experiment_name,
    )


def evaluate_cv(
    estimator,
    X,
    y,
    model,
    config,
    experiment_name=None,
    fit_params=None,
):
    result = cross_validate(
        estimator,
        X,
        y,
        cv=get_cv(),
        scoring=[
            "accuracy",
            "roc_auc",
        ],
        params=fit_params,
    )

    metrics = {
        "accuracy_mean": result["test_accuracy"].mean(),
        "accuracy_std": result["test_accuracy"].std(),
        "roc_auc_mean": result["test_roc_auc"].mean(),
        "roc_auc_std": result["test_roc_auc"].std(),
    }

    if experiment_name is not None:
        save_experiment(
            name=experiment_name,
            model=model,
            config=config,
            metrics=metrics,
        )

    return metrics


def save_experiment(
    name: str,
    model: BaseEstimator | str,
    config: dict,
    metrics: dict,
    path: str = "experiments/experiments.csv",
    model_params: dict | None = None,
) -> None:
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
        "model_params": json.dumps(
            params,
            default=str,
        ),
        "config": json.dumps(
            config,
            default=str,
        ),
        **metrics,
    }

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame([experiment])

    df.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )





def get_cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )



def build_experiment(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    config: ExperimentConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:

    if config is None:
        config = base_config

    X, numeric, binary, onehot = prepare_features(
        df=df,
        config=config,
    )
    
    y = df[target]

    preprocessor = build_preprocessor(
        numeric_features=numeric,
        binary_features=binary,
        onehot_features=onehot,
        config=config,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return X, y, pipeline



def tune_model(
    df: pd.DataFrame,
    target: str,
    model: BaseEstimator,
    param_grid: dict,
    config: ExperimentConfig | None = None,
) -> GridSearchCV:

    X, y, pipeline = build_experiment(
        df=df,
        target=target,
        model=model,
        config=config,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        },
        cv=get_cv(),
        n_jobs=-1,
        refit="accuracy",
        verbose=0
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
    """Save a selected CV row using caller-provided metric and iteration columns."""
    if selection_metric not in metric_columns:
        raise ValueError("selection_metric must be present in metric_columns")

    score_column = metric_columns[selection_metric]
    best_idx = (cv_results[score_column].idxmax() if higher_is_better
                else cv_results[score_column].idxmin())
    row = cv_results.loc[best_idx]

    metrics = {
        metric: row[column]
        for metric, column in metric_columns.items()
    }

    experiment_config = {
        **config,
        "cv_method": cv_method,
        "selection_metric": selection_metric,
    }
    saved_params = params.copy()
    if iteration_column is not None:
        selected_iterations = int(row[iteration_column]) + iteration_offset
        experiment_config["selected_iterations"] = selected_iterations
        if iteration_param is not None:
            experiment_config["cv_max_iterations"] = params.get(iteration_param)
            saved_params[iteration_param] = selected_iterations

    save_experiment(
        name=name,
        model=model_name,
        config=experiment_config,
        metrics=metrics,
        model_params=saved_params,
    )

    return metrics
