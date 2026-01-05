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
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from src.interfaces import DatasetBase, TransformerBase, TransformerHeadBase
from src.utils import build_data_loader, drop_helpers, forward_context

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


class Trainer:
    def __init__(self, cfg: DictConfig, output_dir: Path, disable_logging: bool = False):
        self.cfg = cfg
        self.output_dir = output_dir

        self.log_writer = None
        if not disable_logging:
            self._init_logging()

        self._init_data()
        self._init_model()
        self._init_optim()
        self._init_fabric()
        self._init_checkpoints()

        logger.info(f"Gradient Projection: {'Enabled' if self.cfg.proj_grad else 'Disabled'}")
        logger.info(f"Gradient Mixing: {'Enabled' if self.cfg.mix_grad else 'Disabled'}")

    def _init_logging(self):
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=str(log_dir))

    def _init_data(self):
        self.train_dataset: DatasetBase = instantiate(self.cfg.dataset)
        n_samples_val = int(self.cfg.dataset.n_samples * self.cfg.val_split)

        self.val_datasets: dict[int, DatasetBase] = {}
        self.val_dataloaders: dict[int, DataLoader] = {}

        for n_digits in range(self.cfg.val_min_digits, self.cfg.val_max_digits + 1):
            self.val_datasets[n_digits] = instantiate(
                self.cfg.dataset, max_digits=n_digits, n_samples=n_samples_val, seed=self.cfg.dataset.seed + 1
            )

            self.val_dataloaders[n_digits] = build_data_loader(self.val_datasets[n_digits], self.cfg, train=False)

        self.val_dataset = self.val_datasets[self.cfg.dataset.max_digits]

        logger.info(f"Dataset split: Train ({len(self.train_dataset)}), Val ({len(self.val_dataset)})")

        subset_rng = torch.Generator()
        subset_rng.manual_seed(self.cfg.runtime.seed)
        train_log_dataset = Subset(
            self.train_dataset,
            torch.randperm(len(self.train_dataset), generator=subset_rng)[: self.cfg.logging.n_log_samples],
        )

        self.train_dataloader = build_data_loader(self.train_dataset, self.cfg, train=True)
        self.train_log_dataloader = build_data_loader(train_log_dataset, self.cfg, train=False)

    def _init_model(self):
        model_cfg = OmegaConf.to_container(self.cfg.model, resolve=True)
        drop_helpers(model_cfg)

        self.head: TransformerHeadBase = instantiate(self.cfg.head, dataset=self.train_dataset)
        self.model: TransformerBase = instantiate(model_cfg, head=self.head, dataset=self.train_dataset)

        self.log_model()

    def _init_optim(self):
        self.optim = instantiate(self.cfg.optim, params=[*self.model.parameters(), *self.head.parameters()])

    def _init_fabric(self):
        self.fabric = Fabric(accelerator=self.cfg.runtime.accelerator, devices=self.cfg.runtime.devices)

        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        self.head = self.fabric.setup(self.head)

        self.train_dataloader, self.train_log_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, self.train_log_dataloader
        )

        for k in self.val_dataloaders.keys():
            self.val_dataloaders[k] = self.fabric.setup_dataloaders(self.val_dataloaders[k])

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

        if self.log_writer:
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
            if param.numel() == 0:
                continue

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

    def forward(self, batch: tuple[Tensor, Optional[Tensor]]) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[dict],
    ]:
        with forward_context(orig_shape=batch[0].shape):
            x_in, x_out = batch

            src, tgt = self.model.prepare_tokens(x_in, x_in)  # todo: check if this makes sense
            enc_out, dec_out, enc_attns, dec_self_attns, dec_cross_attns, extra = self.model(src, tgt)

            y_pred = self.head(dec_out if dec_out is not None else enc_out)

            if x_out is not None:
                loss, accuracy = self.head.forward_loss(y_pred, x_in, x_out)
            else:
                loss = accuracy = self.fabric.to_device(torch.tensor(0.0))

            return loss, accuracy, y_pred, enc_out, dec_out, enc_attns, dec_self_attns, dec_cross_attns, extra

    def solve(self, dataset: DatasetBase, epoch: int):
        sample = self.fabric.to_device(dataset.get_example())
        x_in = sample[0:1, ...]

        gt_steps = [dataset.to_string(sample[i]).split("\n") for i in range(sample.shape[0])]
        steps: list[list[str]] = [dataset.to_string(x_in[0]).split("\n")]

        for i in range(len(gt_steps) - 1):
            with torch.no_grad():
                _, _, y_pred, _, _, _, _, _, _ = self.forward((x_in, None))
                out = self.head.step(x_in, y_pred)

            x_in = out
            steps.append(dataset.to_string(x_in[0]).split("\n"))

        gt_str = "\n".join("   ".join(gt_steps[i][j] for i in range(len(gt_steps))) for j in range(len(gt_steps[0])))
        pred_str = "\n".join("   ".join(steps[i][j] for i in range(len(steps))) for j in range(len(steps[0])))
        print("=== Solution ===")
        print(gt_str)
        print("=== Prediction ===")
        print(pred_str)

        if self.log_writer:
            if epoch == 0:
                self.log_writer.add_text("full_example/gt", f"```\n{gt_str}```", epoch)

            self.log_writer.add_text("full_example/pred", f"```\n{pred_str}```", epoch)

    def val_k_epoch(self, k: int, epoch: int) -> tuple[float, float]:
        self.model.eval()
        self.head.eval()

        val_dataset = self.val_datasets[k]
        val_dataloader = self.val_dataloaders[k]
        self.solve(val_dataset, epoch)

        total_loss = 0.0
        total_accuracy = 0.0
        it = tqdm(val_dataloader, unit="batch", desc=f"[{k}] Val Epoch {epoch}/{self.cfg.n_epochs}")
        for i, batch in enumerate(it):
            batch: tuple[Tensor, Tensor]

            with torch.no_grad():
                loss, accuracy, y_pred, _, _, _, _, _, _ = self.forward(batch)  # todo: log outputs

            total_loss += loss.item()
            total_accuracy += accuracy.item()

            if i < self.cfg.logging.n_log_samples:
                x_in, x_out = batch[0][0], batch[1][0]

                x_in_str = val_dataset.to_string(x_in)
                x_out_str = val_dataset.to_string(x_out)

                with torch.no_grad():
                    if type(y_pred) is tuple:
                        y_pred_mapped = tuple(y_p[0:1, ...] for y_p in y_pred)
                        out = self.head.step(x_in[None, ...], y_pred_mapped)
                    else:
                        out = self.head.step(x_in[None, ...], y_pred[0:1, ...])

                out_str = val_dataset.to_string(out[0])

                log_str = ""
                for line_in, gt_line_out, line_out in zip(
                    x_in_str.split("\n")[:-1], x_out_str.split("\n")[:-1], out_str.split("\n")[:-1]
                ):
                    log_str += f"{line_in}    -->    {gt_line_out}    |    {line_out}\n"

                if self.log_writer:
                    self.log_writer.add_text(f"val/{k}_sample_{i}", f"```\n{log_str}```", epoch)

        avg_loss = total_loss / len(val_dataloader)
        avg_accuracy = total_accuracy / len(val_dataloader)

        logger.info(f"[{k}] Epoch {epoch}: Val Loss: {avg_loss:.6f}")
        logger.info(f"[{k}] Epoch {epoch}: Val Accuracy: {100*avg_accuracy:.2f}%")

        if self.log_writer:
            self.log_writer.add_scalar(f"val/{k}_loss", avg_loss, epoch)
            self.log_writer.add_scalar(f"val/{k}_accuracy", avg_accuracy, epoch)

        return avg_loss, avg_accuracy

    def val_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.eval()
        self.head.eval()

        out = None

        for k in self.val_dataloaders.keys():
            k_out = self.val_k_epoch(k, epoch)

            if k == self.cfg.dataset.max_digits:
                out = k_out

        return out

    def extract_gradients(self) -> list[Tensor]:
        grads = []
        params = [*self.model.parameters(), *self.head.parameters()]
        for param in params:
            if param.grad is not None:
                grads.append(param.grad.detach().clone())
            else:
                grads.append(None)
        return grads

    def project_grads(self, train_grads: list[Tensor], val_grads: list[Tensor]):
        params = [*self.model.parameters(), *self.head.parameters()]
        for i, param in enumerate(params):
            g_train = train_grads[i]
            g_val = val_grads[i]
            if g_train is not None and g_val is not None:
                val_normsq = torch.sum(g_val * g_val)
                dot_product = torch.sum(g_train * g_val) / (val_normsq + 1e-16)
                if dot_product < 0:
                    # Remove component of the gradient that points in the opposite direction
                    g_proj = g_train - dot_product * g_val
                else:
                    # Already pointing in same direction
                    g_proj = g_train

                param.grad.copy_(g_proj)

    def train_epoch(self, epoch: int):
        self.model.train()
        self.head.train()

        val_it = iter(self.val_dataloaders[self.cfg.dataset.max_digits + 1])

        it = tqdm(self.train_dataloader, unit="batch", desc=f"Epoch {epoch}/{self.cfg.n_epochs}")
        for i, batch in enumerate(it):
            batch: tuple[Tensor, Tensor]

            self.optim.zero_grad(set_to_none=False)
            loss, accuracy = self.forward(batch)[:2]

            if self.cfg.proj_grad:
                # Project gradients into half-space defined by validation gradient (digit + 1)
                self.fabric.backward(loss)
                grad = self.extract_gradients()
                self.optim.zero_grad(set_to_none=False)

                try:
                    val_batch = next(val_it)
                except StopIteration:
                    val_it = iter(self.val_dataloaders[self.cfg.dataset.max_digits + 1])
                    val_batch = next(val_it)

                val_loss, val_accuracy = self.forward(val_batch)[:2]
                self.fabric.backward(val_loss)

                val_grad = self.extract_gradients()
                self.optim.zero_grad(set_to_none=False)

                self.project_grads(grad, val_grad)
            elif self.cfg.mix_grad:
                # Mix training and validation gradients (digit + 1)
                self.fabric.backward(loss * 0.5)
                try:
                    val_batch = next(val_it)
                except StopIteration:
                    val_it = iter(self.val_dataloaders[self.cfg.dataset.max_digits + 1])
                    val_batch = next(val_it)

                val_loss, val_accuracy = self.forward(val_batch)[:2]
                self.fabric.backward(val_loss * 0.5)
            else:
                # Only use training gradient
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
