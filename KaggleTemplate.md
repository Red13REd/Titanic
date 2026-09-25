# Titanic — experiment summary

### To-do

- Decision Tree: baseline → разбор переобучения → tuning.
- Random Forest: baseline → tuning.
- Сравнить модели на одинаковом 5-fold CV.

### Notes

- Основная метрика: **accuracy**.
- Дополнительно смотрим **ROC-AUC**.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- GridSearch выбирает модель по `refit="accuracy"`.

### Solution

| № | Approach | CV Accuracy | CV STD | Notes |
| --- | --- | ---: | ---: | --- |
| 1 | DummyClassifier | 0.615 | — | baseline |
| 2 | LogisticRegression — base features | 0.7924 | 0.0188 | до tuning |
| 3 | LogisticRegression — base features + tuning | 0.8025 | — | tuning помогает |
| 4 | LogisticRegression — feature engineering | **0.8159** | **0.0081** | лучший результат |
| 5 | LogisticRegression — FE + L1/L2/ElasticNet tuning | **0.8159** | 0.0081 | улучшения нет |

### Logistic Regression — итог

Лучший вариант:

- `C=1`
- L2 (`l1_ratio=0`)
- `solver="saga"` в эксперименте с ElasticNet
- `max_iter=1000`; лучшая модель сошлась за **187 итераций**

Feature engineering поднял accuracy с **0.8025** после tuning базовых признаков до **0.8159** — примерно **+1.35 п.п.**

### Processed ideas

#### Good

- `Pclass` → one-hot
- `FareLog1p`
- `SexPclass`
- `AgeSquared`
- `IsAlone`
- Hyperparameter tuning на базовых признаках: 0.7924 → 0.8025

#### Neutral

- L1 / L2 / ElasticNet: лучший результат остался у L2, прироста поверх 0.8159 нет
- `FamilySize`
- смена encoding для `Sex`
- дополнительное масштабирование само по себе заметного прироста не дало

#### Bad

- `IsChild` снизил среднюю accuracy
- слишком сильная регуляризация заметно ухудшает качество

### Final Ensemble

| Model | CV | Public LB | Private LB |
| --- | ---: | ---: | ---: |
|  |  |  |  |
