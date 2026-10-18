"""Verify parameter-search training never evaluates the test loader."""
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset
from variants.dinov3.trainer import Trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x), x


class ForbiddenTestLoader:
    def __iter__(self):
        raise AssertionError('Parameter search must not evaluate the test set')


class ValidationTests(unittest.TestCase):
    def test_training_saves_selection_without_touching_test(self):
        torch.manual_seed(0)
        model = TinyModel()
        temporal = torch.nn.Linear(3, 3)
        samples = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(samples, labels, samples, samples), batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            Trainer(model, temporal, torch.optim.Adam(model.parameters()), torch.optim.Adam(temporal.parameters()),
                    loader, loader, ForbiddenTestLoader(), torch.device('cpu'), logging.getLogger('test'),
                    SimpleNamespace(num_epoch=1), folder, 'ft_10p', evaluate_test=False)
            selection = json.loads((Path(folder) / 'selection.json').read_text())
            self.assertEqual(selection['epoch'], 1)
            self.assertTrue((Path(folder) / 'saved_models/ckp_best.pt').is_file())
            self.assertTrue((Path(folder) / 'saved_models/resume.pt').is_file())


if __name__ == '__main__':
    unittest.main()
