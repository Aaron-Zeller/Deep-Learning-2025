import logging
import math
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.fabric import Fabric, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from src.interfaces import DatasetBase, TransformerBase, TransformerHeadBase
from src.utils import build_data_loader, drop_helpers

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir

        self._init_logging()
        self._init_data()
        self._init_model()
        self._init_optim()
        self._init_fabric()
        self._init_checkpoints()

    def _init_logging(self):
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(log_dir))

    def _init_data(self):
        self.dataset: DatasetBase = instantiate(self.cfg.dataset)

        split_rng = torch.Generator()
        split_rng.manual_seed(self.cfg.runtime.seed)
        val_size = int(len(self.dataset) * self.cfg.val_split)
        train_size = len(self.dataset) - val_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size], generator=split_rng)
        logger.info(f"Dataset split: Train ({train_size}), Val ({val_size})")

        train_log_dataset = Subset(
            train_dataset, torch.randperm(train_size, generator=split_rng)[: self.cfg.logging.n_log_samples]
        )

        self.train_dataloader = build_data_loader(train_dataset, self.cfg, train=True)
        self.train_log_dataloader = build_data_loader(train_log_dataset, self.cfg, train=False)
        self.val_dataloader = build_data_loader(val_dataset, self.cfg, train=False)

    def _init_model(self):
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        drop_helpers(model_cfg)

        self.model: TransformerBase = instantiate(model_cfg, dataset=self.dataset)
        self.head: TransformerHeadBase = instantiate(self.cfg.head, transformer=self.model, dataset=self.dataset)

        self.log_model()

    def _init_optim(self):
        self.optim = instantiate(self.cfg.optim, params=[*self.model.parameters(), *self.head.parameters()])

    def _init_fabric(self):
        self.fabric = Fabric(accelerator=self.cfg.runtime.accelerator, devices=self.cfg.runtime.devices)

        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        self.head = self.fabric.setup(self.head)

        self.train_dataloader, self.train_log_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.train_log_dataloader, self.val_dataloader
        )

    def _init_checkpoints(self):
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        self.start_epoch = 1
        if self.cfg.resume or self.cfg.resume_ckpt is not None:
            ckpt = Path.cwd() / self.cfg.resume_ckpt if self.cfg.resume_ckpt is not None else None
            if ckpt is None:  # try to find latest checkpoint
                exp_dir = self.output_dir.parent.parent
                ckpts = [*sorted(exp_dir.glob("**/**/checkpoints/*.pth"))]

                if len(ckpts) > 0:
                    ckpt = ckpts[-1]

            if ckpt is not None:
                logger.info(f"Resuming from checkpoint: {ckpt}")
                rest = self.fabric.load(ckpt, dict(model=self.model, head=self.head, optim=self.optim))
                self.start_epoch = rest["epoch"] + 1  # don't train twice
            else:
                logger.info("No checkpoints found, training from scratch.")

    def log_model(self):
        summary_kwargs = dict(col_names=["num_params", "params_percent"], verbose=0)
        model_summary = summary(self.model, depth=4, **summary_kwargs)
        head_summary = summary(self.head, **summary_kwargs)

        logger.info("Model Summary:\n" + str(model_summary))
        logger.info("Head Summary:\n" + str(head_summary))

        self.log_writer.add_text("model_summary", f"```{model_summary}```")
        self.log_writer.add_text("head_summary", f"```{head_summary}```")

    def save_epoch(self, epoch: int):
        epoch_str = str(epoch).zfill(int(math.log10(self.cfg.n_epochs)) + 1)
        ckpt_path = self.ckpt_dir / f"ckpt_{epoch_str}.pth"
        self.fabric.save(ckpt_path, dict(model=self.model, head=self.head, optim=self.optim, epoch=epoch))
        logger.info(f"Epoch {epoch}: Saved checkpoint to {ckpt_path}")

    def log_param_histograms(self, step: int):
        total_param_norm, n_param_norm = 0.0, 0
        total_grad_norm, n_grad_norm = 0.0, 0

        params = [*self.model.named_parameters(), *self.head.named_parameters()]
        for name, param in params:
            w = param.detach()
            total_param_norm += (w**2).sum().item()
            n_param_norm += w.numel()

            self.log_writer.add_histogram(f"{name}/param", w.cpu(), step)

            if param.grad is not None:
                g = param.grad.detach()
                total_grad_norm += (g**2).sum().item()
                n_grad_norm += g.numel()

                self.log_writer.add_histogram(f"{name}/grad", g.cpu(), step)

        avg_param_norm = total_param_norm / n_param_norm
        avg_grad_norm = total_grad_norm / n_grad_norm

        self.log_writer.add_scalar("train/avg_param_norm", avg_param_norm, step)
        self.log_writer.add_scalar("train/avg_grad_norm", avg_grad_norm, step)

    def forward(
        self, batch: tuple[Tensor, Optional[Tensor]]
    ) -> tuple[
        Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]
    ]:
        x_in, x_out = batch

        src, tgt = self.model.prepare_tokens(x_in, x_in, self.head)  # todo: check if this makes sense
        enc_out, dec_out, enc_attns, dec_self_attns, dec_cross_attns = self.model(src, tgt)

        y_pred = self.head(dec_out if dec_out is not None else enc_out)

        if x_out is not None:
            loss, accuracy = self.head.forward_loss(y_pred, x_in, x_out)
        else:
            loss = accuracy = self.fabric.to_device(torch.tensor(0.0))

        return loss, accuracy, y_pred, enc_out, dec_out, enc_attns, dec_self_attns, dec_cross_attns

    def solve(self, epoch: int):
        sample = self.fabric.to_device(self.dataset.get_example())
        x_in = sample[0:1, ...]

        gt_steps = [self.dataset.to_string(sample[i]).split("\n") for i in range(sample.shape[0])]
        steps: list[list[str]] = [self.dataset.to_string(x_in[0]).split("\n")]

        for i in range(len(gt_steps) - 1):
            with torch.no_grad():
                _, _, y_pred, _, _, _, _, _ = self.forward((x_in, None))
                out = self.head.step(x_in, y_pred)

            x_in = out
            steps.append(self.dataset.to_string(x_in[0]).split("\n"))

        gt_str = "\n".join("   ".join(gt_steps[i][j] for i in range(len(gt_steps))) for j in range(len(gt_steps[0])))
        pred_str = "\n".join("   ".join(steps[i][j] for i in range(len(steps))) for j in range(len(steps[0])))
        print("=== Solution ===")
        print(gt_str)
        print("=== Prediction ===")
        print(pred_str)

        if epoch == 0:
            self.log_writer.add_text("full_example/gt", f"```\n{gt_str}```", epoch)

        self.log_writer.add_text("full_example/pred", f"```\n{pred_str}```", epoch)

    def val_epoch(self, epoch: int):
        self.model.eval()
        self.head.eval()

        self.solve(epoch)

        total_loss = 0.0
        total_accuracy = 0.0
        it = tqdm(self.val_dataloader, unit="batch", desc=f"Val Epoch {epoch}/{self.cfg.n_epochs}")
        for i, batch in enumerate(it):
            batch: tuple[Tensor, Tensor]

            with torch.no_grad():
                loss, accuracy, y_pred, _, _, _, _, _ = self.forward(batch)  # todo: log outputs

            total_loss += loss.item()
            total_accuracy += accuracy.item()

            if i < self.cfg.logging.n_log_samples:
                x_in, x_out = batch[0][0], batch[1][0]

                x_in_str = self.dataset.to_string(x_in)
                x_out_str = self.dataset.to_string(x_out)

                with torch.no_grad():
                    out = self.head.step(x_in[None, ...], y_pred[0:1, ...])

                out_str = self.dataset.to_string(out[0])

                log_str = ""
                for line_in, gt_line_out, line_out in zip(
                    x_in_str.split("\n")[:-1], x_out_str.split("\n")[:-1], out_str.split("\n")[:-1]
                ):
                    log_str += f"{line_in}    -->    {gt_line_out}    |    {line_out}\n"

                self.log_writer.add_text(f"val/sample_{i}", f"```\n{log_str}```", epoch)

        avg_loss = total_loss / len(self.val_dataloader)
        avg_accuracy = total_accuracy / len(self.val_dataloader)

        self.log_writer.add_scalar("val/loss", avg_loss, epoch)
        logger.info(f"Epoch {epoch}: Val Loss: {avg_loss:.6f}")

        self.log_writer.add_scalar("val/accuracy", avg_accuracy, epoch)
        logger.info(f"Epoch {epoch}: Val Accuracy: {100*avg_accuracy:.2f}%")

    def train_epoch(self, epoch: int):
        self.model.train()
        self.head.train()

        it = tqdm(self.train_dataloader, unit="batch", desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        for i, batch in enumerate(it):
            batch: tuple[Tensor, Tensor]

            self.optim.zero_grad()
            loss, accuracy = self.forward(batch)[:2]
            self.fabric.backward(loss)
            self.optim.step()

            step = (epoch - 1) * len(self.train_dataloader) + i
            self.log_writer.add_scalar("train/loss", loss.item(), step)
            self.log_writer.add_scalar("train/accuracy", accuracy.item(), step)

            it.set_postfix(dict(loss=loss.item(), accuracy=f"{100*accuracy.item():.2f}%"))

            self.log_writer.add_scalar("train/lr", self.optim.param_groups[0]["lr"], step)
            self.log_writer.add_scalar("train/epoch", epoch, step)

            if step % self.cfg.logging.param_hist_every_n_steps == 0:
                self.log_param_histograms(step)

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
        logger.info("Enabled TF32 (Tensor Cores). Turn this off if there are precision issues.")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Writing outputs to: {output_dir}")

    trainer = Trainer(cfg, output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
