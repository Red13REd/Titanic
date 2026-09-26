# Titanic — experiment summary

### To-do

- CatBoost
- LightGBM
- XGBoost
- После основных моделей: отдельный этап feature engineering для `Cabin`, `Name`, `Ticket` и других признаков.

### Notes

- Основная метрика: **accuracy**.
- Дополнительно смотрим **ROC-AUC**.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- GridSearch выбирает модель по `refit="accuracy"`.
- Scaling важен для KNN; для деревьев и Random Forest не нужен.

### Solution

| № | Approach | CV Accuracy | CV STD | Notes |
| --- | --- | ---: | ---: | --- |
| 1 | DummyClassifier | 0.6150 | — | baseline |
| 2 | LogisticRegression — base features | 0.7924 | 0.0188 | до tuning |
| 3 | LogisticRegression — base features + tuning | 0.8025 | — | tuning помогает |
| 4 | LogisticRegression — feature engineering | 0.8159 | 0.0081 | лучший LR |
| 5 | KNN — baseline | 0.8047 | 0.0225 | default + scaling |
| 6 | KNN — tuned + feature engineering | 0.8317 | 0.0086 | лучший KNN |
| 7 | Decision Tree — без ограничений | 0.7879 | 0.0311 | сильное переобучение |
| 8 | Decision Tree — tuned + `SexPclass` | 0.8429 | 0.0202 | лучший DT |
| 9 | Random Forest — baseline | 0.8125 | 0.0272 | train ≈ 0.98 |
| 10 | **Random Forest — tuned** | **0.8462** | — | лучший результат |

### Logistic Regression — итог

Лучший вариант:

- `C=1`
- L2
- feature engineering
- **accuracy = 0.8159**

Tuning базовых признаков дал **0.8025**, feature engineering поднял результат до **0.8159**.

### KNN — итог

Лучшие параметры:

- `n_neighbors=21`
- `weights="uniform"`
- `metric="manhattan"`

Scaling + feature selection / engineering:

**0.8047 → 0.8148 → 0.8317**

### Decision Tree — итог

Неограниченное дерево:

- train: **0.9798**
- CV: **0.7879**

После tuning:

- `criterion="gini"`
- `max_depth=8`
- `min_samples_split=10`
- `min_samples_leaf=1`

Лучший feature experiment: `SexPclass`.

**CV accuracy: 0.8429 ± 0.0202**

### Random Forest — итог

Baseline:

- train: **0.9798**
- CV: **0.8125 ± 0.0272**
- ROC-AUC: **0.8717 ± 0.0267**

После tuning:

- `n_estimators=200`
- `max_depth=None`
- `max_features=None`
- `min_samples_leaf=4`
- `min_samples_split=10`
- train: **0.8923**
- CV: **0.8462**

Feature engineering существенного прироста не дал.

### Processed ideas

#### Good

- Logistic Regression: `Pclass` → one-hot, `FareLog1p`, `SexPclass`, `AgeSquared`, `IsAlone`
- KNN: scaling, `n_neighbors=21`, Manhattan distance, `FamilySize`, feature selection
- Decision Tree: ограничение сложности, `SexPclass`
- Random Forest: `min_samples_leaf=4`, `min_samples_split=10`
- Random Forest: 200 деревьев дали стабильный tuned ensemble

#### Neutral

- Logistic Regression: L1 / L2 / ElasticNet не улучшили лучший результат
- KNN: добавление всех engineered features сразу не помогло
- Decision Tree: `entropy`, scaling и большинство дополнительных фич не улучшили лучший результат
- Random Forest: `FamilySize`, `IsAlone`, `SexPclass`, смена encoding не дали заметного прироста
- Random Forest: `sqrt` / `log2` для `max_features` не обошли `None`

#### Bad

- Logistic Regression: слишком сильная регуляризация ухудшает качество
- KNN без scaling заметно хуже
- Decision Tree без ограничений: сильное переобучение
- Random Forest baseline: train **0.9798** при CV **0.8125**

### Final Ensemble

| Model | CV | Public LB | Private LB |
| --- | ---: | ---: | ---: |
|  |  |  |  |
