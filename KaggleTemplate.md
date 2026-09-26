# Titanic — experiment summary

### To-do

- Random Forest: baseline → tuning.
- Сравнить лучшие модели на одинаковом 5-fold CV.

### Notes

- Основная метрика: **accuracy**.
- Дополнительно смотрим **ROC-AUC**.
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- GridSearch выбирает модель по `refit="accuracy"`.
- Для KNN scaling важен; для Decision Tree scaling не нужен.

### Solution

| № | Approach | CV Accuracy | CV STD | Notes |
| --- | --- | --- | --- | --- |
| 1 | DummyClassifier | 0.6150 | — | baseline |
| 2 | LogisticRegression — base features | 0.7924 | 0.0188 | до tuning |
| 3 | LogisticRegression — base features + tuning | 0.8025 | — | tuning помогает |
| 4 | LogisticRegression — feature engineering | 0.8159 | 0.0081 | лучший LR |
| 5 | KNN — baseline | 0.8047 | 0.0225 | default + scaling |
| 6 | KNN — tuned base features | 0.8148 | 0.0106 | k=21, uniform, Manhattan |
| 7 | KNN — tuned + feature engineering | 0.8317 | 0.0086 | лучший KNN |
| 8 | Decision Tree — без ограничений | 0.7879 | 0.0311 | сильное переобучение |
| 9 | Decision Tree — tuned | 0.8384 | \~0.018 | depth=8, split=10 |
| 10 | **Decision Tree — tuned + SexPclass** | **0.8429** | **0.0202** | лучший результат |

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

Лучший набор включал scaling, `FamilySize`, one-hot для `Sex` и исключение `Fare`.

**Accuracy: 0.8047 → 0.8148 → 0.8317**

### Decision Tree — итог

Неограниченное дерево:

- train accuracy: **0.9798**
- CV accuracy: **0.7879**

После ограничения сложности:

- `criterion="gini"`
- `max_depth=8`
- `min_samples_split=10`
- `min_samples_leaf=1`
- train accuracy: **0.8822**
- CV accuracy: **0.8384**

Лучший feature experiment: `SexPclass`.

**Итоговый CV accuracy: 0.8429 ± 0.0202**\
**ROC-AUC: 0.8472 ± 0.0212**

### Processed ideas

#### Good

- Logistic Regression: `Pclass` → one-hot, `FareLog1p`, `SexPclass`, `AgeSquared`, `IsAlone`
- Logistic Regression: tuning базовых признаков
- KNN: scaling
- KNN: `n_neighbors=21`, uniform weights, Manhattan distance
- KNN: `FamilySize`, one-hot для `Sex`, исключение `Fare`
- Decision Tree: ограничение `max_depth`
- Decision Tree: `min_samples_split=10`
- Decision Tree: `SexPclass`

#### Neutral

- Logistic Regression: L1 / L2 / ElasticNet не дали прироста поверх 0.8159
- KNN: добавление всех engineered features сразу не помогло
- Decision Tree: `criterion="entropy"` не улучшил результат
- Decision Tree: scaling не дал пользы
- Decision Tree: `Pclass` one-hot дал небольшой, но не лучший результат
- Decision Tree: `FareLog1p`, `IsAlone`, `IsChild`, `AgeSquared` не улучшили лучший результат

#### Bad

- Logistic Regression: слишком сильная регуляризация ухудшает качество
- KNN без scaling: заметное падение accuracy
- Decision Tree без ограничений: **train 0.9798 vs CV 0.7879**
- Decision Tree: `FamilySize` ухудшил результат

### Final Ensemble

| Model | CV | Public LB | Private LB |
| --- | --- | --- | --- |
|  |  |  |  |
