from pathlib import Path
import logging
import math

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader, Subset
from hydra.utils import instantiate
from lightning.fabric import Fabric, seed_everything
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary
from tqdm import tqdm

from src.interfaces import DatasetBase, TransformerBase
from src.utils import drop_helpers

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        # =================== #
        # ===== Logging ===== #
        # =================== #
        log_dir = output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(log_dir))

        # ================ #
        # ===== Data ===== #
        # ================ #
        dataset: DatasetBase = instantiate(cfg.dataset)

        split_rng = torch.Generator()
        split_rng.manual_seed(cfg.runtime.seed)
        val_size = int(len(dataset) * cfg.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_rng)
        log.info(f"Dataset split: Train ({train_size}), Val ({val_size})")

        train_log_dataset = Subset(
            train_dataset, torch.randperm(train_size, generator=split_rng)[: cfg.logging.n_log_samples]
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.runtime.n_workers,
            pin_memory=cfg.runtime.pin_memory,
        )

        self.train_log_dataloader = DataLoader(
            train_log_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.runtime.n_workers,
            pin_memory=cfg.runtime.pin_memory,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.runtime.n_workers,
            pin_memory=cfg.runtime.pin_memory,
        )

        # ================= #
        # ===== Model ===== #
        # ================= #
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        drop_helpers(model_cfg)

        self.model: TransformerBase = instantiate(model_cfg, vocab_size=dataset.vocab_size())

        self.log_model()

        # ================== #
        # ===== Optim ====== #
        # ================== #
        self.optim = instantiate(cfg.optim, params=self.model.parameters())

        # ================== #
        # ===== Fabric ===== #
        # ================== #
        self.fabric = Fabric(accelerator=cfg.runtime.accelerator, devices=cfg.runtime.devices)
        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        self.train_dataloader, self.train_log_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.train_log_dataloader, self.val_dataloader
        )

        # ====================== #
        # ===== Checkpoints ==== #
        # ====================== #
        self.ckpt_dir = output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.start_epoch = 1
        if cfg.resume:
            ckpts = [*sorted(self.ckpt_dir.glob("*.pth"))]
            if len(ckpts) > 0:
                ckpt = ckpts[-1]
                log.info(f"Resuming from checkpoint: {ckpt}")

                rest = self.fabric.load(ckpt, dict(model=self.model, optim=self.optim))
                self.start_epoch = rest["epoch"] + 1  # don't train twice
            else:
                log.info("No checkpoints found, training from scratch.")

    def log_model(self):
        summary(self.model)

    def save_epoch(self, epoch: int):
        epoch_str = str(epoch).zfill(int(math.log10(self.cfg.n_epochs)) + 1)
        ckpt_path = self.ckpt_dir / f"ckpt_{epoch_str}.pth"
        self.fabric.save(ckpt_path, dict(model=self.model, optim=self.optim, epoch=epoch))
        logging.info(f"Epoch {epoch}: Saved checkpoint to {ckpt_path}")

    def val_epoch(self, epoch: int):
        # todo
        pass

    def train_epoch(self, epoch: int):
        # todo
        pass

    def run(self):
        if self.start_epoch == 1:
            self.save_epoch(0)  # Initial checkpoint before training
            self.val_epoch(0)  # Initial validation before training

        for epoch in range(self.start_epoch, self.cfg.n_epochs + 1):  # Indexing starts at 1
            self.train_epoch(epoch)

            if epoch % self.cfg.logging.save_every_n_epochs == 0 or epoch == self.cfg.n_epochs:
                self.save_epoch(epoch)

            if epoch % self.cfg.logging.val_every_n_epochs == 0 or epoch == self.cfg.n_epochs:
                self.val_epoch(epoch)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.runtime.seed)

    if cfg.runtime.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Check for TF32 support
    if torch.cuda.is_available() and torch.cuda.is_tf32_supported() and cfg.runtime.allow_tf32:
        torch.set_float32_matmul_precision("high")  # this is "highest" otherwise
        log.info("Enabled TF32 (Tensor Cores). Turn this off if there are precision issues.")

    output_dir = Path(HydraConfig.get().runtime.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Writing outputs to: {output_dir}")

    trainer = Trainer(cfg, output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
