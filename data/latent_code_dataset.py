"""
Dataset for field function
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data


class LatentCodeDataset(data.Dataset):

    def __init__(self, input_dir, subset_samples=None):

        assert os.path.exists(input_dir)
        self.latent_codes = np.load(
            os.path.join(input_dir, 'selected_latent_code.npy')).astype(float)
        self.labels = np.load(
            os.path.join(input_dir, 'selected_pred_class.npy')).astype(int)
        self.scores = np.load(
            os.path.join(input_dir, 'selected_pred_scores.npy')).astype(float)

        self.latent_codes = torch.FloatTensor(self.latent_codes)
        self.labels = torch.LongTensor(self.labels)
        self.scores = torch.FloatTensor(self.scores)

        # select a subset from train set
        if subset_samples is not None and len(
                self.latent_codes) > subset_samples:
            idx = list(range(len(self.latent_codes)))
            selected_idx = random.sample(idx, subset_samples)
            self.latent_codes = [self.latent_codes[i] for i in selected_idx]
            self.labels = [self.labels[i] for i in selected_idx]
            self.scores = [self.scores[i] for i in selected_idx]

        assert len(self.latent_codes) == len(self.labels)
        assert len(self.labels) == len(self.scores)

    def __getitem__(self, index):
        return (self.latent_codes[index], self.labels[index],
                self.scores[index])

    def __len__(self):
        return len(self.latent_codes)
