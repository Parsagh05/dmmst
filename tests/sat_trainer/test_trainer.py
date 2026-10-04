"""Tests for the Trainer class.

SATTrainer subclasses the Hugging Face ``Trainer``, so these tests drive it end to end
with a tiny real model instead of mocking the training loop.
"""

import os
import shutil
import unittest
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers.utils import ModelOutput

from sat.transformers.trainer import SATTrainer as Trainer
from sat.transformers.trainer import TrainingArgumentsWithMPSSupport


@dataclass
class _Output(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

    def forward(self, input_ids, labels=None):
        logits = self.linear(input_ids)
        loss = torch.nn.functional.mse_loss(logits, labels)
        return _Output(loss=loss, logits=logits)


class _DictDataset(Dataset):
    def __init__(self, n):
        g = torch.Generator().manual_seed(n)
        self.x = torch.randn(n, 5, generator=g)
        self.y = torch.randn(n, 2, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"input_ids": self.x[i], "labels": self.y[i]}


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "./test-trainer-output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.model = _TinyModel()
        self.args = TrainingArgumentsWithMPSSupport(
            output_dir=self.test_output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=1e-2,
            fp16=False,
            gradient_accumulation_steps=1,
            seed=42,
            report_to=[],
            use_cpu=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=_DictDataset(10),
            eval_dataset=_DictDataset(6),
        )

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_train(self):
        before = self.model.linear.weight.detach().clone()
        result = self.trainer.train()
        self.assertEqual(result.global_step, 5)  # 10 samples / batch 2
        self.assertFalse(torch.equal(before, self.model.linear.weight.detach()))

    def test_evaluate(self):
        metrics = self.trainer.evaluate()
        self.assertIn("eval_loss", metrics)
        self.assertGreater(metrics["eval_loss"], 0.0)

    def test_predict(self):
        out = self.trainer.predict(_DictDataset(4))
        self.assertEqual(out.predictions.shape, (4, 2))
        self.assertEqual(out.label_ids.shape, (4, 2))

    def test_save_model(self):
        self.trainer.save_model(self.test_output_dir)
        self.assertTrue(
            any(
                f.startswith(("model.safetensors", "pytorch_model"))
                for f in os.listdir(self.test_output_dir)
            )
        )

    def test_training_arguments_mps_support(self):
        self.assertEqual(self.args.device.type, "cpu")
        self.assertEqual(self.args.local_process_index, 0)


if __name__ == "__main__":
    unittest.main()
