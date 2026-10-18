import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        X_train = torch.as_tensor(X_train)
        y_train = torch.as_tensor(y_train, dtype=torch.long)
        if X_train.ndim == 2:
            X_train = X_train.unsqueeze(1)
        if X_train.ndim != 3:
            raise ValueError('Expected samples [N, C, T] or [N, T, C]')
        if X_train.shape[1] != config.input_channels:
            if X_train.shape[2] != config.input_channels:
                raise ValueError('Channel count does not match config')
            X_train = X_train.permute(0, 2, 1)
        self.x_data = X_train.float()
        self.y_data = y_train.long()

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size


    if "_1p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"), map_location="cpu", weights_only=True)
    elif "_5p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"), map_location="cpu", weights_only=True)
    elif "_10p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"), map_location="cpu", weights_only=True)
    elif "_50p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"), map_location="cpu", weights_only=True)
    elif "_75p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"), map_location="cpu", weights_only=True)
    elif training_mode == "SupCon":
        train_dataset = torch.load(os.path.join(data_path, "pseudo_train_data.pt"), map_location="cpu", weights_only=True)
    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"), map_location="cpu", weights_only=True)

    valid_dataset = torch.load(os.path.join(data_path, "val.pt"), map_location="cpu", weights_only=True)
    test_dataset = torch.load(os.path.join(data_path, "test.pt"), map_location="cpu", weights_only=True)

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, 'supervised')
    test_dataset = Load_Dataset(test_dataset, configs, 'supervised')

    if train_dataset.__len__() < batch_size:
        batch_size = train_dataset.__len__()

    contrastive = training_mode in ('self_supervised', 'SupCon')
    if batch_size < 2:
        raise ValueError('Training needs at least two labeled samples for BatchNorm. Use a larger label budget.')
    # Never discard validation/test or pseudo-label tail samples.
    drop_train = contrastive or (len(train_dataset) % batch_size == 1 and training_mode != 'gen_pseudo_labels')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=training_mode != 'gen_pseudo_labels', drop_last=drop_train, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False, drop_last=False)
    return train_loader, valid_loader, test_loader
