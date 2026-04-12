"""
Condition loaders for class-conditional models (ImageNet).
"""

import torch


class ImageNetIterator:
    """Yields (cond_labels, uncond_labels) for class-conditional generation."""

    def __init__(self, model, scale, batch_size,
                 num_samples_per_class=50, n_classes=1000, device='cpu'):
        self.model = model
        self.scale = scale
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.n_classes = n_classes
        self.current_value = 0
        self.current_num_cls_sample = 0
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        batch = [self.current_value] * self.batch_size
        self.current_num_cls_sample += self.batch_size
        if self.current_num_cls_sample >= self.num_samples_per_class:
            self.current_value = (self.current_value + 1) % self.n_classes
            self.current_num_cls_sample = 0

        cond_labels = torch.LongTensor(batch).to(self.device)

        if self.scale != 1.0:
            uncond_labels = torch.full_like(cond_labels, self.n_classes)
        else:
            uncond_labels = None

        return cond_labels, uncond_labels
