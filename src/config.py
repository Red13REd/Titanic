"""Чтение настроек запуска и создание выбранной модели."""

from importlib import import_module
from pathlib import Path

import yaml


MODELS = {
    "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
    "decision_tree": ("sklearn.tree", "DecisionTreeClassifier"),
    "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
    "knn": ("sklearn.neighbors", "KNeighborsClassifier"),
    "catboost": ("catboost", "CatBoostClassifier"),
    "lightgbm": ("lightgbm", "LGBMClassifier"),
    "xgboost": ("xgboost", "XGBClassifier"),
}


def load_config(path="config.yaml"):
    """Читает общий YAML; пути в нём задаются относительно корня проекта."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Общий конфиг должен быть словарём настроек")
    return config


def create_model(model_config, seed):
    """Создаёт классификатор; явно заданный random_state важнее общего seed."""
    name = model_config["name"]
    if name not in MODELS:
        raise ValueError(f"Неизвестная модель {name!r}; доступны {tuple(MODELS)}")
    module_name, class_name = MODELS[name]
    constructor = getattr(import_module(module_name), class_name)
    parameters = dict(model_config.get("params", {}))
    if name != "knn":
        parameters.setdefault("random_state", seed)
    return constructor(**parameters)
