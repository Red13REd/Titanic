"""Запуск выбранной модели: CV, обучение на train и сабмит для Kaggle."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import create_model, load_config
from src.preprocessing import prepare_features
from src.sklearn_experiments import build_experiment


def main(config_path="config.yaml"):
    """Читает config_path, оценивает модель на CV и обучает на всём train.

    Возвращает средние CV-метрики и std. В output_dir заменяются metrics.json,
    submission.csv от модели на всём train и submission_<N>fold.csv от N моделей CV.
    Пути данных задаются относительно рабочей папки.
    """
    config = load_config(config_path)
    data = config["data"]
    cv = StratifiedKFold(
        **config["cv"],
        random_state=config["seed"] if config["cv"]["shuffle"] else None,
    )
    train = pd.read_csv(data["train_path"])
    model = create_model(config["model"], config["seed"])
    preprocessing_config = config["preprocessing"]
    X, y, pipeline = build_experiment(
        df=train.drop(columns=data["drop_columns"]),
        target=data["target"],
        model=model,
        config=preprocessing_config,
    )
    scores = cross_validate(
        pipeline, X, y, cv=cv, scoring=["accuracy", "roc_auc"],
        error_score="raise", return_estimator=True,
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
    test_X, _, _, _ = prepare_features(
        test.drop(columns=data["drop_columns"]), preprocessing_config,
    )
    predictions = pipeline.predict(test_X)
    submission = pd.DataFrame({data["id_column"]: test[data["id_column"]], data["target"]: predictions})

    # Каждый pipeline применяет обработку, обученную только на своём train-фолде.
    fold_probabilities = [
        fold_model.predict_proba(test_X)[:, list(fold_model.classes_).index(1)]
        for fold_model in scores["estimator"]
    ]
    mean_probabilities = np.mean(fold_probabilities, axis=0)
    cv_submission = pd.DataFrame({
        data["id_column"]: test[data["id_column"]],
        data["target"]: (mean_probabilities >= 0.5).astype(int),
    })

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False)
    cv_filename = f"submission_{len(scores['estimator'])}fold.csv"
    cv_submission.to_csv(output_dir / cv_filename, index=False)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Results: {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
