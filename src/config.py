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
    """Читает YAML из path и возвращает словарь настроек без изменения путей."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Общий конфиг должен быть словарём настроек")
    return config


def create_model(model_config, seed):
    """Возвращает необученный классификатор по model_config[name, params].

    seed задаёт random_state, если он не указан явно; KNN seed не использует.
    """
    name = model_config["name"]
    if name not in MODELS:
        raise ValueError(f"Неизвестная модель {name!r}; доступны {tuple(MODELS)}")
    module_name, class_name = MODELS[name]
    constructor = getattr(import_module(module_name), class_name)
    parameters = dict(model_config.get("params", {}))
    if name != "knn":
        parameters.setdefault("random_state", seed)
    return constructor(**parameters)
