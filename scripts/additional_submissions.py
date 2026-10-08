"""Сабмиты выбранных EmbeddingMLP и XGBoost: весь train и пять фолдов.

Запуск из корня: .venv/Scripts/python.exe -X utf8 scripts/additional_submissions.py
DNN использует конфигурацию и лучшие эпохи из сохранённой внешней CV.
Нового выбора параметров или эпох по Kaggle нет; RF-файлы не изменяются.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dnn_training import EmbeddingMLP, predict, resolve_device, train_one_epoch
from src.preprocessing import base_config, prepare_features
from src.sklearn_experiments import build_experiment


def embedding_probabilities(train_X, train_y, frames, settings, epochs, device):
    """Обучает свежую сеть на train_X/y; возвращает вероятности frames.

    Предобработка fit только на train_X. Эпохи заранее фиксированы,
    frames — список неразмеченных DataFrame в нужном порядке строк.
    """
    numeric = settings["features"]["numeric"]
    categorical = settings["features"]["categorical"]
    preprocessing = settings["preprocessing"]
    numeric_steps = [SimpleImputer(strategy=preprocessing["numeric_imputer_strategy"])]
    if preprocessing["scale_numeric"]:
        numeric_steps.append(StandardScaler())
    preprocessor = ColumnTransformer([
        ("numeric", make_pipeline(*numeric_steps), numeric),
        ("categorical", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-2,
            encoded_missing_value=-1,
        ), categorical),
    ])
    train_values = preprocessor.fit_transform(train_X)
    cardinalities = [
        len(values) + 2
        for values in preprocessor.named_transformers_["categorical"].categories_
    ]

    def tensors(values):
        """Разделяет результат обработки на float32 числа и int64 индексы."""
        if not np.isfinite(values).all():
            raise ValueError("После предобработки остались нечисловые значения")
        numbers = values[:, preprocessor.output_indices_["numeric"]]
        categories = values[:, preprocessor.output_indices_["categorical"]] + 2
        for column, size in enumerate(cardinalities):
            if not ((categories[:, column] >= 0) & (categories[:, column] < size)).all():
                raise ValueError("Индекс категории выходит за границы Embedding")
        return torch.tensor(numbers, dtype=torch.float32), torch.tensor(categories, dtype=torch.long)

    inputs = tensors(train_values)
    targets = torch.tensor(train_y.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model_config = settings["model"]
    model = EmbeddingMLP(
        model_config, inputs[0].shape[1], cardinalities,
        embedding_dim=model_config["embedding_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **settings["optimizer"]["params"])
    batch_size = settings["loader"]["batch_size"]
    loader = DataLoader(TensorDataset(*inputs, targets), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        loss = train_one_epoch(model, loader, nn.BCEWithLogitsLoss(), optimizer)
        if not np.isfinite(loss):
            raise ValueError("DNN получила нечисловой train loss")
    probabilities = []
    for frame in frames:
        loader = DataLoader(
            TensorDataset(*tensors(preprocessor.transform(frame))),
            batch_size=batch_size, shuffle=False,
        )
        probabilities.append(torch.sigmoid(predict(model, loader, has_targets=False)).numpy().ravel())
    return probabilities


def main():
    """Создаёт четыре CSV, вероятности и описание запусков в отдельной папке."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    data = config["data"]
    train = pd.read_csv(PROJECT_ROOT / data["train_path"])
    test = pd.read_csv(PROJECT_ROOT / data["test_path"])
    y = train[data["target"]]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(train, y))
    cv_rows = pd.read_csv(PROJECT_ROOT / "experiments/dnn_cv_results.csv")
    cv_rows = cv_rows.loc[cv_rows["candidate"] == "embedding_24_32_16_8"].sort_values("fold")
    saved_splits = json.loads((PROJECT_ROOT / "experiments/dnn_cv_splits.json").read_text(encoding="utf-8"))
    if cv_rows["fold"].tolist() != [1, 2, 3, 4, 5] or saved_splits["rows"] != len(train):
        raise ValueError("Нужны сохранённые результаты пяти DNN-фолдов для текущих данных")
    for (development, outer), saved in zip(splits, saved_splits["splits"], strict=True):
        if not np.array_equal(outer, saved["outer"]) or sorted(development) != sorted(saved["train"] + saved["validation"]):
            raise ValueError("Сохранённые DNN-фолды не совпадают с разбиением seed 42")
    snapshots = [json.loads(value) for value in cv_rows["config_json"]]
    if any(snapshot != snapshots[0] for snapshot in snapshots) or snapshots[0]["seed"] != 42:
        raise ValueError("Настройки DNN должны совпадать на всех пяти фолдах")
    settings = snapshots[0]["dnn"]
    device = resolve_device(config["dnn"].get("device", "cpu"))
    torch.set_num_threads(1)
    best_epochs = cv_rows["best_epoch"].astype(int).tolist()
    single_epochs = int(np.median(best_epochs))

    xgb_params = dict(
        n_estimators=80, learning_rate=0.1, max_depth=4, min_child_weight=5,
        random_state=42, eval_metric="logloss", n_jobs=1,
    )
    preprocessing = {"features": base_config["features"].copy(), "scale_numeric": False}
    X, _, pipeline = build_experiment(train, data["target"], XGBClassifier(**xgb_params), preprocessing)
    test_X, _, _, _ = prepare_features(test, preprocessing)
    all_probabilities = {}
    pipeline.fit(X, y)
    all_probabilities["xgboost"] = pipeline.predict_proba(test_X)[:, 1]
    all_probabilities["embedding"] = embedding_probabilities(
        train, y, [test], settings, single_epochs, device,
    )[0]
    print(f"Обучены одиночные модели; DNN: {single_epochs} эпох", flush=True)

    fold_results = []
    fold_scores = {"xgboost": [], "embedding": []}
    for fold, ((development, outer), epochs) in enumerate(zip(splits, best_epochs, strict=True), 1):
        model = clone(pipeline).fit(X.iloc[development], y.iloc[development])
        fold_scores["xgboost"].append(model.predict_proba(test_X)[:, 1])
        outer_scores = {"xgboost": model.predict_proba(X.iloc[outer])[:, 1]}
        test_scores, outer_scores["embedding"] = embedding_probabilities(
            train.iloc[development], y.iloc[development],
            [test, train.iloc[outer]], settings, epochs, device,
        )
        fold_scores["embedding"].append(test_scores)
        for name, scores in outer_scores.items():
            fold_results.append({
                "model": name, "fold": fold, "train_rows": len(development),
                "epochs": epochs if name == "embedding" else None,
                "accuracy": accuracy_score(y.iloc[outer], scores >= 0.5),
                "roc_auc": roc_auc_score(y.iloc[outer], scores),
            })
        print(f"Фолд {fold}/5 готов; DNN: {epochs} эпох", flush=True)
    for name, scores in fold_scores.items():
        all_probabilities[f"{name}_5fold"] = np.mean(scores, axis=0)

    output = PROJECT_ROOT / "experiments/runs/additional_submissions"
    output.mkdir(parents=True, exist_ok=True)
    for name, probabilities in all_probabilities.items():
        if probabilities.shape != (len(test),) or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError(f"{name}: неверные вероятности")
        pd.DataFrame({
            data["id_column"]: test[data["id_column"]],
            data["target"]: (probabilities >= 0.5).astype(int),
        }).to_csv(output / f"submission_{name}.csv", index=False)
    pd.DataFrame({data["id_column"]: test[data["id_column"]], **all_probabilities}).to_csv(output / "probabilities.csv", index=False)
    pd.DataFrame(fold_results).to_csv(output / "fold_metrics.csv", index=False)
    metadata = {
        "seed": 42, "threshold": 0.5, "device": str(device),
        "embedding": settings, "single_epochs": single_epochs, "fold_epochs": best_epochs,
        "epoch_source": "experiments/dnn_cv_results.csv; inner validation, no outer fold",
        "xgboost": {"params": xgb_params, "preprocessing": preprocessing},
        "protocol": "Fresh refit on full train or each 80% development fold; mean test probabilities. Fold metrics describe individual models, not the five-model ensemble.",
    }
    (output / "run_settings.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Сабмиты: {output}")


if __name__ == "__main__":
    main()
