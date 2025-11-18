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
