"""Запуск выбранной модели: CV, обучение на train и сабмит для Kaggle."""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config import create_model, load_config


def categorical_values(frame):
    """Приводит категории к строкам, сохраняя пропуски для заполнения."""
    values = frame.astype("string").astype(object)
    return values.where(values.notna(), np.nan)


def main(config_path="config.yaml"):
    """Запускает решение из YAML; файлы результатов в output_dir заменяются."""
    config = load_config(config_path)
    data = config["data"]
    cv = StratifiedKFold(
        **config["cv"],
        random_state=config["seed"] if config["cv"]["shuffle"] else None,
    )
    train = pd.read_csv(data["train_path"])
    X = train.drop(columns=[data["target"], *data["drop_columns"]])
    y = train[data["target"]]
    model = create_model(config["model"], config["seed"])
    categorical = Pipeline([
        ("strings", FunctionTransformer(categorical_values, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
         ["Age", "SibSp", "Parch", "Fare"]),
        ("categorical", categorical, ["Pclass", "Sex", "Embarked"]),
    ], remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=["accuracy", "roc_auc"], error_score="raise",
    )
    result = {
        "name": config["model"]["name"],
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
    }

    pipeline.fit(X, y)
    test = pd.read_csv(data["test_path"])
    predictions = pipeline.predict(test.drop(columns=data["drop_columns"]))
    submission = pd.DataFrame({data["id_column"]: test[data["id_column"]], data["target"]: predictions})

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Results: {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
