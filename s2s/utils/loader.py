"""
Freethreaded dataloader wrapping m00nny_utils Prefetcher.
"""
import sys
import os
from typing import Callable, Iterator, Any, Optional

import torch

_m00nny_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "m00nny_utils")
if _m00nny_path not in sys.path:
    sys.path.insert(0, os.path.dirname(_m00nny_path))

try:
    from m00nny_utils.util.prefetcher_th import Prefetcher
    _prefetcher_available = True
except ImportError:
    _prefetcher_available = False
    Prefetcher = None


class _FallbackDataLoader:
    """Simple dataloader fallback when m00nny_utils is not available."""
    def __init__(self, dataset, transform, batch_size, shuffle=True):
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        batch = []
        for idx in indices:
            sample = self.dataset[idx]
            if self.transform is not None:
                sample = self.transform(sample)
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch:
            yield self._collate(batch)

    def _collate(self, batch):
        keys = batch[0].keys()
        result = {}
        for k in keys:
            vals = [b[k] for b in batch]
            if isinstance(vals[0], torch.Tensor):
                result[k] = torch.stack(vals)
            else:
                result[k] = vals
        return result


class S2SDataLoader:
    """Freethreaded dataloader for S2S training.

    Args:
        dataset: Sequence-like dataset.
        transform: Callable that maps a sample to {"audio": Tensor, "text": str, "labels": Tensor}.
        batch_size: Batch size.
        num_workers: Number of prefetch workers.
        shuffle: Whether to shuffle indices.
    """

    def __init__(
        self,
        dataset,
        transform: Callable[[Any], dict],
        batch_size: int,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        if _prefetcher_available and num_workers > 0:
            self._prefetcher = Prefetcher(num_workers=num_workers)
            self._prefetcher.load(
                dataset=dataset,
                transform=transform,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        else:
            self._prefetcher = None
            self._fallback = _FallbackDataLoader(dataset, transform, batch_size, shuffle)

    def __len__(self) -> int:
        if self._prefetcher is not None:
            return len(self._prefetcher)
        return len(self._fallback)

    def __iter__(self) -> Iterator[dict]:
        if self._prefetcher is not None:
            return iter(self._prefetcher)
        return iter(self._fallback)
