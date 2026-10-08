"""Короткие проверки DNN без записи в основной журнал экспериментов.

Запуск: .venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -v
"""

import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dnn_training import EmbeddingMLP, evaluate, predict, train_model, train_one_epoch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks/dnn.ipynb").read_text(encoding="utf-8"))


def run_cell(index, namespace):
    source = "".join(NOTEBOOK["cells"][index]["source"])
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    exec(compile(source, f"dnn.ipynb:cell-{index}", "exec"), namespace)


def prepare(mode):
    namespace = {}
    run_cell(1, namespace)
    namespace["display"] = lambda *args: None
    namespace["dnn_config"]["preprocessing"]["categorical_encoder"] = mode
    run_cell(3, namespace)
    run_cell(4, namespace)
    return namespace


def short_run(namespace):
    namespace["dnn_config"]["training"]["max_epochs"] = 2
    namespace["run_config"]["dnn"] = copy.deepcopy(namespace["dnn_config"])
    namespace["config_json"] = json.dumps(namespace["run_config"], sort_keys=True)
    namespace["config_id"] = hashlib.sha256(namespace["config_json"].encode()).hexdigest()[:12]
    run_cell(6, namespace)
    return namespace["model"]


class DNNWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Маленькие CPU-батчи не требуют большого пула потоков.
        torch.set_num_threads(1)

    def test_both_modes_training_prediction_and_reproducibility(self):
        for mode in ("one-hot", "embedding"):
            with self.subTest(mode=mode):
                ns = prepare(mode)
                model = short_run(ns)
                metrics = evaluate(model, ns["val_loader"], ns["eval_criterion"])
                self.assertAlmostEqual(metrics["loss"], ns["best_model"]["loss"])
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                self.assertEqual(len(ns["lr_history"]), 2)
                batch = next(iter(ns["val_loader"]))
                self.assertEqual(batch[0].dtype, torch.float32)
                self.assertEqual(batch[-1].dtype, torch.float32)
                if mode == "embedding":
                    self.assertEqual(batch[1].dtype, torch.long)
                self.assertEqual(model(*batch[:-1]).shape, (16, 1))
                self.assertEqual(model(*(value[:1] for value in batch[:-1])).shape, (1, 1))

                labelled = predict(model, ns["val_loader"])
                inputs = ns["val_dataset"].tensors[:-1]
                unlabelled_loader = DataLoader(TensorDataset(*inputs), batch_size=16)
                unlabelled = predict(model, unlabelled_loader, has_targets=False)
                self.assertEqual(labelled.shape, (179, 1))
                self.assertTrue(torch.equal(labelled, unlabelled))
                if mode == "one-hot":
                    raw = predict(model, DataLoader(inputs[0], batch_size=16))
                    self.assertTrue(torch.equal(labelled, raw))
                else:
                    probe = ns["val_X"].iloc[:1].copy()
                    probe["Embarked"] = "__UNKNOWN_PORT__"
                    values = ns["preprocessor"].transform(probe)
                    indices = values[:, ns["categorical_slice"]] + 2
                    self.assertEqual(indices[0, 2], 0)
                    unknown_logits = model(
                        torch.tensor(values[:, ns["numeric_slice"]], dtype=torch.float32),
                        torch.tensor(indices, dtype=torch.long),
                    )
                    self.assertTrue(torch.isfinite(unknown_logits).all())
                    self.assertEqual(ns["mlp_input_features"], 10)

                first_state = copy.deepcopy(model.state_dict())
                first_history = list(ns["train_loss_history"])
                short_run(ns)
                self.assertEqual(first_history, ns["train_loss_history"])
                for key, value in first_state.items():
                    self.assertTrue(torch.equal(value, ns["model"].state_dict()[key]), key)

    def test_embedding_weights_are_trained(self):
        ns = prepare("embedding")
        model = EmbeddingMLP(ns["dnn_config"]["model"], 4, ns["categorical_cardinalities"])
        before = [layer.weight.detach().clone() for layer in model.embeddings]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        train_one_epoch(model, ns["train_loader"], nn.BCEWithLogitsLoss(), optimizer)
        for old, layer in zip(before, model.embeddings):
            self.assertFalse(torch.equal(old, layer.weight))

    def test_invalid_configuration_and_shapes(self):
        ns = prepare("one-hot")
        ns["dnn_config"]["preprocessing"]["categorical_encoder"] = "embeding"
        with self.assertRaisesRegex(ValueError, "categorical_encoder"):
            run_cell(3, ns)
        cfg = ns["dnn_config"]["model"]
        for sizes, dimension in (([0, 4], 2), ([], 2), ([5, 4], 0)):
            with self.assertRaises(ValueError):
                EmbeddingMLP(cfg, 4, sizes, dimension)
        model = EmbeddingMLP(cfg, 4, [5, 4, 6])
        with self.assertRaisesRegex(ValueError, "категориальных"):
            model(torch.zeros(2, 4), torch.zeros(2, 2, dtype=torch.long))
        broken_loader = DataLoader(TensorDataset(torch.zeros(2, 4), torch.zeros(2)))
        with self.assertRaisesRegex(ValueError, "форму"):
            evaluate(nn.Linear(4, 1), broken_loader, nn.BCEWithLogitsLoss())
        binary_model = nn.Linear(4, 1)
        with self.assertRaisesRegex(ValueError, "форму"):
            train_one_epoch(
                binary_model, broken_loader, nn.BCEWithLogitsLoss(),
                torch.optim.SGD(binary_model.parameters(), lr=0.01),
            )

    def test_cosine_stops_at_minimum(self):
        model = nn.Linear(2, 1)
        loader = DataLoader(TensorDataset(torch.zeros(2, 2), torch.tensor([[0.], [1.]])))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0.00003)
        result = train_model(
            model=model, train_loader=loader, val_loader=loader,
            criterion=nn.BCEWithLogitsLoss(), eval_criterion=nn.BCEWithLogitsLoss(),
            optimizer=optimizer, scheduler=scheduler, epoch=6,
        )
        history = result[4]
        self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))
        self.assertTrue(np.allclose(history[3:], 0.00003))
        self.assertEqual(scheduler.last_epoch, 3)

    def test_journal_preserves_legacy_rows_and_updates_same_id(self):
        original = ROOT / "experiments/dnn_results.csv"
        original_bytes = original.read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.csv"
            path.write_bytes(original_bytes)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                legacy = list(csv.DictReader(stream))
            for mode in ("one-hot", "embedding"):
                ns = prepare(mode)
                ns["dnn_config"]["results_path"] = str(path)
                short_run(ns)
                run_cell(10, ns)
                count = len(ns["experiment_results"])
                run_cell(10, ns)
                self.assertEqual(len(ns["experiment_results"]), count)
                new = ns["experiment_result"]
                self.assertEqual(new["categorical_encoder"], mode)
                self.assertEqual(new["mlp_input_features"], 13 if mode == "one-hot" else 10)
                self.assertEqual(new["pos_weight"], 1)
                self.assertEqual(new["eval_loss_fn"], "BCEWithLogitsLoss")
                self.assertEqual(new["model"], "MLP" if mode == "one-hot" else "EmbeddingMLP")
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    saved = list(csv.DictReader(stream))
                for old, current in zip(legacy, saved):
                    for key, value in old.items():
                        self.assertEqual(current[key], value, key)
                    self.assertEqual(current["categorical_encoder"], old.get("categorical_encoder", ""))
            # Веса используются только для обучения; в журнал попадает обычная BCE.
            ns["dnn_config"]["training"]["loss_fn_params"] = {"pos_weight": 2}
            short_run(ns)
            run_cell(10, ns)
            self.assertEqual(ns["experiment_result"]["pos_weight"], 2)
            self.assertIsNone(ns["eval_criterion"].pos_weight)
            self.assertAlmostEqual(ns["best_validation_metrics"]["loss"], ns["best_model"]["loss"])
        self.assertEqual(original.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
