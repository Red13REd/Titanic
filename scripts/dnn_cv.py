"""Пять внешних фолдов для двух зафиксированных DNN-кандидатов.

Запуск из корня проекта: .venv/Scripts/python.exe -X utf8 scripts/dnn_cv.py
Внешний фолд не участвует в предобработке, early stopping или выборе эпохи.
Архитектуры, Adam, LR и seed зафиксированы для сравнения; данные, устройство
и условия early stopping берутся из config.yaml.
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# При запуске файла Python добавляет в sys.path папку scripts, а не корень.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dnn_training import EmbeddingMLP, create_mlp, evaluate, predict, resolve_device, train_model
from src.config import load_config


CANDIDATES = {
    "one_hot_24": {"encoder": "one-hot", "hidden_layers": [24], "dropout": 0.0},
    "embedding_24_32_16_8": {
        "encoder": "embedding", "hidden_layers": [24, 32, 16, 8], "dropout": 0.15,
    },
}


def prepare_fold(X, y, indices, candidate, settings):
    """Готовит TensorDataset для train, validation и внешнего фолда.

    indices задаёт позиции строк, candidate — кодирование категорий,
    settings — признаки и предобработку. Fit выполняется только на train.
    Возвращает datasets и размеры Embedding-таблиц (для one-hot — None).
    Сдвиг индексов на 2 резервирует 0 для неизвестных и 1 для пропусков.
    """
    numeric = settings["features"]["numeric"]
    categorical = settings["features"]["categorical"]
    preprocessing = settings["preprocessing"]
    numeric_steps = [SimpleImputer(strategy=preprocessing["numeric_imputer_strategy"])]
    if preprocessing["scale_numeric"]:
        numeric_steps.append(StandardScaler())
    if candidate["encoder"] == "one-hot":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown=preprocessing["handle_unknown"])
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1,
        )
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(*numeric_steps), numeric),
        ("categorical", encoder, categorical),
    ])
    transformed = {
        "train": preprocessor.fit_transform(X.iloc[indices["train"]]),
        "validation": preprocessor.transform(X.iloc[indices["validation"]]),
        "outer": preprocessor.transform(X.iloc[indices["outer"]]),
    }
    cardinalities = None
    if candidate["encoder"] == "embedding":
        cardinalities = [len(values) + 2 for values in preprocessor.named_transformers_["categorical"].categories_]
    datasets = {}
    for part, values in transformed.items():
        assert np.isfinite(values).all()
        labels = torch.tensor(y.iloc[indices[part]].to_numpy(), dtype=torch.float32).reshape(-1, 1)
        if candidate["encoder"] == "one-hot":
            datasets[part] = TensorDataset(torch.tensor(values, dtype=torch.float32), labels)
        else:
            numbers = values[:, preprocessor.output_indices_["numeric"]]
            category_indices = values[:, preprocessor.output_indices_["categorical"]] + 2
            assert np.equal(category_indices, np.floor(category_indices)).all()
            for column, size in enumerate(cardinalities):
                assert ((category_indices[:, column] >= 0) & (category_indices[:, column] < size)).all()
            datasets[part] = TensorDataset(
                torch.tensor(numbers, dtype=torch.float32),
                torch.tensor(category_indices, dtype=torch.long), labels,
            )
    return datasets, cardinalities


def _build_splits(X, y):
    """Возвращает пять внешних фолдов с внутренней validation, seed=42."""
    splits = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (development, outer) in enumerate(splitter.split(X, y), 1):
        train, validation = train_test_split(
            development, test_size=0.2, random_state=42, stratify=y.iloc[development],
        )
        assert not set(train) & set(validation)
        assert not set(train) & set(outer)
        assert not set(validation) & set(outer)
        assert len(train) + len(validation) + len(outer) == len(X)
        splits.append({
            "fold": fold, "train": train.tolist(),
            "validation": validation.tolist(), "outer": outer.tolist(),
        })
    coverage = [index for split in splits for index in split["outer"]]
    assert sorted(coverage) == list(range(len(X)))
    return splits


def _run_fold(X, y, source, split, name, candidate, config, device):
    """Обучает кандидата, восстанавливает лучшие веса и оценивает внешний фолд.

    Возвращает строку метрик и предсказания пассажиров; файлы не записывает.
    """
    started = time.perf_counter()
    settings = config["dnn"]
    seed = 42
    datasets, cardinalities = prepare_fold(X, y, split, candidate, settings)
    model_config = {
        "hidden_layers": candidate["hidden_layers"], "activation": "ReLU",
        "batch_norm": False, "dropout": candidate["dropout"], "embedding_dim": 2,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if candidate["encoder"] == "one-hot":
        model = create_mlp(model_config, datasets["train"].tensors[0].shape[1])
    else:
        model = EmbeddingMLP(
            model_config, datasets["train"].tensors[0].shape[1], cardinalities,
            embedding_dim=2,
        )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0)
    loaders = {
        part: DataLoader(dataset, batch_size=16, shuffle=part == "train", num_workers=0)
        for part, dataset in datasets.items()
    }
    training = settings["training"]
    training_result = train_model(
        model=model, train_loader=loaders["train"], val_loader=loaders["validation"],
        criterion=nn.BCEWithLogitsLoss(), eval_criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer, scheduler=None, epoch=training["max_epochs"],
        min_epoch=training["min_epochs"], patience=training["patience"],
        min_delta=training["min_delta"], threshold=0.5,
    )
    inner_metrics = evaluate(model, loaders["validation"], nn.BCEWithLogitsLoss())
    assert np.isclose(inner_metrics["loss"], training_result[-1]["loss"])
    outer_metrics = evaluate(model, loaders["outer"], nn.BCEWithLogitsLoss())
    scores = torch.sigmoid(predict(model, loaders["outer"])).numpy().reshape(-1)
    assert np.isfinite(scores).all()
    run_settings = copy.deepcopy(settings)
    run_settings["model"] = model_config
    run_settings["preprocessing"]["categorical_encoder"] = candidate["encoder"]
    run_settings["optimizer"] = {"name": "Adam", "params": {"lr": 0.003, "weight_decay": 0}}
    run_settings["scheduler"]["name"] = None
    run_settings["loader"].update(batch_size=16, train_shuffle=True, validation_shuffle=False, num_workers=0)
    run_settings["training"].pop("loss_fn_params", None)
    run_settings["training"]["loss_fn"] = "BCEWithLogitsLoss"
    run_settings["evaluation"]["threshold"] = 0.5
    row = {
        "candidate": name, "fold": split["fold"], "seed": seed,
        "device": str(device),
        "train_rows": len(split["train"]), "validation_rows": len(split["validation"]),
        "outer_rows": len(split["outer"]), "best_epoch": training_result[-1]["epoch"],
        "epochs_run": len(training_result[0]), "inner_validation_loss": inner_metrics["loss"],
        "outer_loss": outer_metrics["loss"], "outer_accuracy": outer_metrics["accuracy"],
        "outer_roc_auc": outer_metrics["roc_auc"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.perf_counter() - started,
        "config_json": json.dumps({"seed": seed, "dnn": run_settings}, sort_keys=True),
    }
    fold_predictions = []
    for index, probability in zip(split["outer"], scores):
        fold_predictions.append({
            "candidate": name, "fold": split["fold"], "row_index": index,
            "PassengerId": source.iloc[index][config["data"]["id_column"]],
            "target": int(y.iloc[index]), "probability": float(probability),
            "prediction": int(probability >= 0.5),
        })
    return row, fold_predictions


def _summarize_results(results, predictions, row_count):
    """Собирает сводку из строк метрик и OOF-предсказаний для row_count объектов.

    Выбор: максимальная средняя Accuracy, затем меньший loss и размер модели.
    Возвращает словарь с победителем, метриками и различиями между фолдами.
    """
    frame = pd.DataFrame(results)
    prediction_frame = pd.DataFrame(predictions)
    summary = []
    for name, group in frame.groupby("candidate"):
        oof = prediction_frame.query("candidate == @name")
        assert len(oof) == row_count and oof["row_index"].nunique() == row_count
        summary.append({
            "candidate": name,
            "accuracy_mean": group["outer_accuracy"].mean(),
            "accuracy_std": group["outer_accuracy"].std(ddof=1),
            "loss_mean": group["outer_loss"].mean(),
            "roc_auc_mean": group["outer_roc_auc"].mean(),
            "oof_accuracy": accuracy_score(oof["target"], oof["prediction"]),
            "oof_roc_auc": roc_auc_score(oof["target"], oof["probability"]),
            "oof_loss": log_loss(oof["target"], oof["probability"]),
            "parameters_min": int(group["parameters"].min()),
            "parameters_max": int(group["parameters"].max()),
            "best_epochs": group["best_epoch"].tolist(),
        })
    winner = min(
        summary,
        key=lambda candidate: (
            -candidate["accuracy_mean"], candidate["loss_mean"], candidate["parameters_max"],
        ),
    )
    payload = {
        "selection_rule": "max mean outer accuracy; ties: min mean loss, then fewer parameters",
        "winner": winner["candidate"], "summary": summary,
        "fold_accuracy_difference_embedding_minus_one_hot": (
            frame.pivot(index="fold", columns="candidate", values="outer_accuracy")
            .eval("embedding_24_32_16_8 - one_hot_24").tolist()
        ),
        "limitation": "Candidates were previously selected using this dataset; not an independent final test.",
    }
    return payload


def run_cv():
    """Запускает 10 обучений, заменяет CSV/JSON в experiments и возвращает сводку."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    settings = config["dnn"]
    device = resolve_device(settings.get("device", "cpu"))
    print(f"DNN device: {device}", flush=True)
    torch.set_num_threads(1)
    source = pd.read_csv(PROJECT_ROOT / config["data"]["train_path"])
    features = settings["features"]
    X = source[features["categorical"] + features["numeric"]].copy()
    y = source[config["data"]["target"]]
    X[features["categorical"]] = X[features["categorical"]].fillna(
        settings["preprocessing"]["categorical_missing_value"],
    )
    output = PROJECT_ROOT / "experiments"
    output.mkdir(exist_ok=True)
    splits = _build_splits(X, y)
    (output / "dnn_cv_splits.json").write_text(
        json.dumps({"split_seed": 42, "source": config["data"]["train_path"],
                    "rows": len(X), "splits": splits}, indent=2), encoding="utf-8",
    )

    results, predictions = [], []
    for split in splits:
        for name, candidate in CANDIDATES.items():
            row, fold_predictions = _run_fold(
                X, y, source, split, name, candidate, config, device,
            )
            results.append(row)
            predictions.extend(fold_predictions)
            pd.DataFrame(results).to_csv(output / "dnn_cv_results.csv", index=False)
            pd.DataFrame(predictions).to_csv(output / "dnn_cv_predictions.csv", index=False)
            print(
                f"{name} fold={split['fold']} accuracy={row['outer_accuracy']:.5f} "
                f"loss={row['outer_loss']:.5f} auc={row['outer_roc_auc']:.5f} "
                f"best_epoch={row['best_epoch']} epochs={row['epochs_run']}", flush=True,
            )

    payload = _summarize_results(results, predictions, len(X))
    (output / "dnn_cv_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    run_cv()
