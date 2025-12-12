import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.addition import AdditionDataset
from src.datasets.multiplication import MultiplicationDataset
from src.datasets.subtraction import SubtractionDataset
from src.interfaces import DatasetBase


class EpochAwareDatasetWrapper(DatasetBase):
    """
    Wraps multiple datasets and switches or merges them depending on the epoch.

    Behavior:
      - merge_mode = False:
            epoch_ranges: [0, 5, 20]
            0-4  -> dataset 0
            5-19 -> dataset 1
            20+  -> dataset 2

      - merge_mode = True:
            0-4  -> dataset 0
            5-19 -> dataset 0 U dataset 1
            20+  -> dataset 0 U dataset 1 U dataset 2

    Dataset merging is implemented via index routing, NOT ConcatDataset,
    to preserve dataset attributes and custom methods.
    """

    def __init__(self, datasets, epoch_ranges, n_samples=None, seed=None, merge_mode: bool = True):

        assert len(epoch_ranges) == len(datasets), "epoch_ranges must match dataset list length"

        self.datasets = datasets
        self.epoch_ranges = epoch_ranges
        self.merge_mode = merge_mode

        self.current_epoch = 0
        self.n_samples = n_samples
        self.seed = seed
        self.token_to_idx = list("0123456789+ -_*\n")

        self.h, self.w = 0, 0
        for ds in self.datasets:
            h, w = ds.grid_size()
            self.h = max(self.h, h)
            self.w = max(self.w, w)
        for d in datasets:
            d.token_to_idx = self.token_to_idx
            d.h = self.h
            d.w = self.w

        # Precompute cumulative lengths for merged index routing
        self._update_active()

    # ---------------------------------------------------------------------
    # Epoch control
    # ---------------------------------------------------------------------

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self._update_active()

    def _get_dataset_index(self) -> int:
        """Return highest index where epoch_ranges[i] <= current_epoch."""
        return max(i for i, e in enumerate(self.epoch_ranges) if self.current_epoch >= e)

    def _update_active(self):
        """Recompute active dataset or merged cumulative boundaries."""
        idx = self._get_dataset_index()

        if not self.merge_mode:
            # Single active dataset
            self.active_datasets = [self.datasets[idx]]
            self.cum_lengths = [0, len(self.datasets[idx])]
            return

        # Merge: include all up to idx → datasets[:idx+1]
        self.active_datasets = self.datasets[: idx + 1]

        # Precompute cumulative lengths
        self.cum_lengths = [0]
        for ds in self.active_datasets:
            self.cum_lengths.append(self.cum_lengths[-1] + len(ds))

    # ---------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, global_idx: int):
        """Route global index into correct underlying dataset."""
        # Find dataset d where cum_lengths[d] <= idx < cum_lengths[d+1]
        for d in range(len(self.active_datasets)):
            if self.cum_lengths[d] <= global_idx < self.cum_lengths[d + 1]:
                local_idx = global_idx - self.cum_lengths[d]
                return self.active_datasets[d][local_idx]

        raise IndexError("Index outside dataset boundaries")

    def get_example(self):
        return self.active_datasets[-1].get_example()

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def to_string(self, x: Tensor) -> str:
        """
        Route to correct sub-dataset depending on shape.
        """
        h, w = x.shape
        mult_index = self.token_to_idx.index("*")
        plus_index = self.token_to_idx.index("+")
        op = "-"
        if mult_index in x:
            op = "*"
        elif plus_index in x:
            op = "+"

        for ds in self.datasets:
            if (
                (ds is AdditionDataset and op != "+")
                or (ds is SubtractionDataset and op != "-")
                or (ds is MultiplicationDataset and op != "*")
            ):
                continue
            if getattr(ds, "h", None) == h and getattr(ds, "w", None) == w:
                return ds.to_string(x)
        return "Invalid Tensor"

    def grid_size(self):
        """
        Return max grid size over active datasets. Be cautious using this, it could break stuff.
        """
        h_max, w_max = 0, 0
        for ds in self.active_datasets:
            h, w = ds.grid_size()
            h_max = max(h_max, h)
            w_max = max(w_max, w)
        return h_max, w_max
