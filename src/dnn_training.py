"""Обучение и оценка бинарных MLP с one-hot или Embedding-входами."""

import copy

import torch
from sklearn.metrics import roc_auc_score
from torch import nn


def resolve_device(name="cpu"):
    """Выбирает cpu, cuda или auto; явный cuda требует доступной GPU."""
    if name not in ("cpu", "cuda", "auto"):
        raise ValueError("dnn.device должен быть 'cpu', 'cuda' или 'auto'")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна: нужна совместимая NVIDIA GPU и сборка PyTorch с CUDA. "
            "Выберите dnn.device: cpu или auto."
        )
    return torch.device(name)


def _check_binary_shapes(logits, targets):
    """Исключает broadcasting меток при бинарной классификации."""
    if logits.ndim != 2 or logits.shape[1] != 1 or logits.shape != targets.shape:
        raise ValueError(
            "Логиты и метки должны иметь одинаковую форму [B, 1]; "
            f"получены {tuple(logits.shape)} и {tuple(targets.shape)}"
        )


def train_one_epoch(model, data_loader, criterion, optimizer):
    """Обновляет веса на размеченном loader; возвращает средний loss.

    Батч: (X, y) или (numeric, categorical, y), метки имеют форму [B, 1].
    criterion вычисляет средний loss батча; optimizer обновляет параметры.
    """
    model.train()
    device = next(model.parameters()).device
    criterion.to(device)
    loss_sum = 0.0
    samples_seen = 0

    for batch in data_loader:
        batch_inputs = tuple(tensor.to(device) for tensor in batch[:-1])
        batch_y = batch[-1].to(device)
        optimizer.zero_grad()
        logits = model(*batch_inputs)
        _check_binary_shapes(logits, batch_y)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.shape[0]
        loss_sum += loss.item() * batch_size
        samples_seen += batch_size

    return loss_sum / samples_seen


def evaluate(model, data_loader, criterion, threshold=0.5):
    """Оценивает размеченный loader без обновления весов.

    Возвращает словарь loss, accuracy и ROC-AUC по всем объектам.
    threshold задаёт порог вероятности для accuracy; ROC-AUC использует скоры.
    """
    model.eval()
    device = next(model.parameters()).device
    criterion.to(device)
    loss_sum = 0.0
    samples_seen = 0
    correct_predictions = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch_inputs = tuple(tensor.to(device) for tensor in batch[:-1])
            batch_y = batch[-1].to(device)
            logits = model(*batch_inputs)
            _check_binary_shapes(logits, batch_y)
            loss = criterion(logits, batch_y)
            scores = torch.sigmoid(logits)
            predicted_classes = scores >= threshold

            batch_size = batch_y.shape[0]
            loss_sum += loss.item() * batch_size
            samples_seen += batch_size
            correct_predictions += (predicted_classes == batch_y).sum().item()
            all_scores.append(scores.cpu())
            all_labels.append(batch_y.cpu())

    scores = torch.cat(all_scores, dim=0).reshape(-1).numpy()
    labels = torch.cat(all_labels, dim=0).reshape(-1).numpy()

    return {
        "loss": loss_sum / samples_seen,
        "accuracy": correct_predictions / samples_seen,
        "roc_auc": roc_auc_score(labels, scores),
    }


def predict(model, data_loader, *, has_targets=True):
    """Возвращает logits [N, 1] на CPU в порядке обхода loader.

    В tuple/list метки находятся последними при has_targets=True.
    Для неразмеченного TensorDataset укажи has_targets=False.
    Батч в виде одного тензора всегда считается одним входом без меток.
    Входы переносятся на устройство модели; результаты готовы для NumPy.
    """
    model.eval()
    device = next(model.parameters()).device
    all_logits = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                batch_inputs = batch[:-1] if has_targets else batch
            else:
                batch_inputs = (batch,)
            batch_inputs = tuple(tensor.to(device) for tensor in batch_inputs)
            all_logits.append(model(*batch_inputs).cpu())

    return torch.cat(all_logits, dim=0)


def train_model(
    *,
    model,
    train_loader,
    val_loader,
    criterion,
    eval_criterion,
    optimizer,
    scheduler=None,
    epoch=200,
    min_epoch=20,
    patience=25,
    min_delta=0.0001,
    verbose=False,
    threshold=0.5,
):
    """Обучает модель, восстанавливает лучший checkpoint и возвращает истории.

    Результат: train loss, validation loss, validation accuracy, validation
    ROC-AUC, LR каждой эпохи и best_model с ключами model (state_dict), loss, epoch.
    criterion используется для обучения, eval_criterion — для выбора checkpoint
    и early stopping. Метки в каждом батче находятся последними.
    CosineAnnealingLR снижается до eta_min, затем LR удерживается постоянным.
    Optimizer остаётся в состоянии последней эпохи; для нового эксперимента
    нужно создать новую модель и optimizer.
    """
    if epoch < 1:
        raise ValueError("Число эпох должно быть положительным")

    epochs_without_improvement = 0
    last_significant_loss = float("inf")
    train_loss_history = []
    validation_loss_history = []
    validation_accuracy_history = []
    validation_roc_auc_history = []
    lr_history = []
    best_model = {"model": None, "loss": float("inf"), "epoch": 0}

    for epoch_index in range(epoch):
        lr_history.append(optimizer.param_groups[0]["lr"])
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        metrics = evaluate(model, val_loader, eval_criterion, threshold=threshold)
        validation_loss = metrics["loss"]
        epoch_number = epoch_index + 1

        train_loss_history.append(train_loss)
        validation_loss_history.append(validation_loss)
        validation_accuracy_history.append(metrics["accuracy"])
        validation_roc_auc_history.append(metrics["roc_auc"])

        if scheduler is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()
            elif scheduler.last_epoch < scheduler.T_max:
                scheduler.step()

        if verbose:
            print(f"Epoch {epoch_number} train_loss_history: {train_loss}")
            print(f"Epoch {epoch_number} validation_loss_history: {validation_loss}")
            print(f"Epoch {epoch_number} validation_accuracy_history: {metrics['accuracy']}")
            print(f"Epoch {epoch_number} validation_roc_auc_score: {metrics['roc_auc']}")

        if validation_loss < best_model["loss"]:
            best_model["loss"] = validation_loss
            best_model["epoch"] = epoch_number
            best_model["model"] = copy.deepcopy(model.state_dict())

        # Сохраняем любой минимум, но patience сбрасываем лишь при улучшении на min_delta.
        if last_significant_loss - validation_loss >= min_delta:
            last_significant_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch_number >= min_epoch and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_model["model"])
    return (
        train_loss_history,
        validation_loss_history,
        validation_accuracy_history,
        validation_roc_auc_history,
        lr_history,
        best_model,
    )


def create_mlp(model_config, in_features):
    """Создаёт MLP из model_config для in_features входов и одного логита.

    Конфиг задаёт hidden_layers, activation, batch_norm и dropout.
    Sigmoid не добавляется: выход подходит для BCEWithLogitsLoss.
    """
    layers = []
    previous_size = in_features

    for hidden_size in model_config["hidden_layers"]:
        layers.append(nn.Linear(previous_size, hidden_size))
        if model_config["batch_norm"]:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(getattr(nn, model_config["activation"])())
        layers.append(nn.Dropout(p=model_config["dropout"]))
        previous_size = hidden_size

    layers.append(nn.Linear(previous_size, 1))
    return nn.Sequential(*layers)


class EmbeddingMLP(nn.Module):
    """Отдельные Embedding для категорий и MLP для объединённых признаков."""

    def __init__(
        self, model_config, numeric_features_count, categorical_cardinalities,
        embedding_dim=2,
    ):
        """Задаёт число числовых входов, размеры таблиц и векторов категорий.

        categorical_cardinalities содержит число строк каждой таблицы,
        включая зарезервированные индексы неизвестных и пропущенных значений.
        """
        super().__init__()
        if not categorical_cardinalities or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in categorical_cardinalities
        ):
            raise ValueError("categorical_cardinalities должен содержать положительные целые размеры")
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim <= 0:
            raise ValueError("embedding_dim должен быть положительным целым числом")

        self.numeric_features_count = numeric_features_count
        in_features = numeric_features_count + len(categorical_cardinalities) * embedding_dim
        self.mlp = create_mlp(model_config, in_features)
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim)
            for num_embeddings in categorical_cardinalities
        ])

    def forward(self, numeric_features, categorical_indices):
        """Объединяет числа [B, F] и категории int64 [B, C] в логиты [B, 1]."""
        if categorical_indices.ndim != 2 or categorical_indices.shape[1] != len(self.embeddings):
            raise ValueError("Число категориальных столбцов должно совпадать с числом Embedding-таблиц")
        if numeric_features.ndim != 2 or numeric_features.shape[1] != self.numeric_features_count:
            raise ValueError("Число числовых столбцов не совпадает с настройками модели")
        if numeric_features.shape[0] != categorical_indices.shape[0]:
            raise ValueError("Числовые признаки и категории должны иметь одинаковый размер батча")

        embedded_features = []
        for index, embedding in enumerate(self.embeddings):
            embedded_feature = embedding(categorical_indices[:, index])
            embedded_features.append(embedded_feature)
        categorical_embeddings = torch.cat(embedded_features, dim=-1)

        x = torch.cat([numeric_features, categorical_embeddings], dim=-1)
        return self.mlp(x)
