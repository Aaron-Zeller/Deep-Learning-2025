from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def drop_helpers(cfg: dict) -> None:
    """Remove helper keys (those starting with '$') from the configuration dictionary.
    This is necessary cause otherwise Hydra will fail to instantiate the objects,
    since these helper variables are not part of the constructor arguments.

    Args:
        cfg: Configuration dictionary to clean.
    """
    for key in list(cfg.keys()):
        if key.startswith("$"):
            del cfg[key]


def build_data_loader(dataset: Dataset, cfg: DictConfig, train: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.runtime.n_workers,
        pin_memory=cfg.runtime.pin_memory,
    )


# Context variable to pass metadata down to all submodules
_forward_metadata: ContextVar[Optional[dict]] = ContextVar("forward_metadata", default=None)


@contextmanager
def forward_context(**metadata):
    """Context manager to set metadata accessible anywhere in the call stack."""
    token = _forward_metadata.set(metadata)
    try:
        yield
    finally:
        _forward_metadata.reset(token)


def get_forward_metadata(key: str, default: Any = None) -> Any:
    """Retrieve metadata from the current context."""
    metadata = _forward_metadata.get()
    return metadata.get(key, default) if metadata else default
