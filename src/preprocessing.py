"""Признаки Titanic и сборка sklearn-преобразований по типам колонок."""

from typing import Literal, NotRequired, TypedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ProcessingType = Literal["numeric", "binary", "onehot", "_"]


class ExperimentConfig(TypedDict):
    """Типы исходных колонок, флаги признаков Titanic и масштабирование чисел."""

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
    """Готовит признаки Titanic из df по config, не изменяя исходный DataFrame.

    Возвращает X и три списка колонок: numeric, binary, onehot.
    Префикс _ или тип _ отключает вход; флаги добавляют производные признаки.
    """
    if config is None:
        config = base_config

    numeric_features: list[str] = []
    binary_features: list[str] = []
    onehot_features: list[str] = []

    active_features = {
        feature: processing_type
        for feature, processing_type in config["features"].items()
        if not feature.startswith("_") and processing_type != "_"
    }

    selected_features = list(active_features.keys())
    required_features = selected_features.copy()

    # Исходные колонки нужны для производных признаков даже при отключённом входе.
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

    for feature, processing_type in active_features.items():
        if processing_type == "numeric":
            numeric_features.append(feature)
        elif processing_type == "binary":
            binary_features.append(feature)
        elif processing_type == "onehot":
            onehot_features.append(feature)
        else:
            raise ValueError(f"Unknown processing type: {processing_type}")

    if config.get("family_size", False):
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        numeric_features.append("FamilySize")
        selected_features.append("FamilySize")

    if config.get("is_alone", False):
        X["IsAlone"] = (X["SibSp"] + X["Parch"] == 0).astype(int)

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
        X["SexPclass"] = X["Sex"].astype(str) + "_" + X["Pclass"].astype(str)

        onehot_features.append("SexPclass")
        selected_features.append("SexPclass")

    return (
        X[selected_features],
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
    """Возвращает необученный ColumnTransformer для переданных списков колонок.

    Числа: медиана и необязательный StandardScaler; категории: one-hot, неизвестные
    значения игнорируются. config управляет масштабированием, fit вызывается позже.
    """

    if config is None:
        config = base_config

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.get("scale_numeric", True):
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_features),
        ("binary", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_features),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_features),
    ])
