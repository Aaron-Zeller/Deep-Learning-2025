import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Callable
from tqdm import tqdm

import torch
from omegaconf import OmegaConf
from tabulate import tabulate
import logging

from train import Trainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# To add a new metric, define a function here and add it to the METRIC_REGISTRY.
# Signature: (trainer, k_idx, epoch) -> float
def metric_step_accuracy(trainer: Trainer, k: int, epoch: int = 0) -> float:
    """Wrapper for the existing per-step validation."""
    _, acc = trainer.val_epoch(0)
    return acc


def metric_full_sample_accuracy(trainer: Trainer, k: int, epoch: int = 0) -> float:
    """Wrapper for full step accuracy over entire sample sequence."""
    return trainer.val_full_sample_epoch(k=k, epoch=epoch)


def metric_full_sample_off_by_one_accuracy(trainer: Trainer, k: int, epoch: int = 0) -> float:
    """Wrapper for full step accuracy allowing off-by-one errors."""
    trainer.model.eval()
    trainer.head.eval()
    val_dataset = trainer.val_datasets[k]
    n_samples = len(val_dataset.data)
    correct_count = 0

    # unrolling steps
    steps_to_rollout = val_dataset.seq_len - 1

    pbar = tqdm(range(n_samples), desc=f"[{k}] Rollout Epoch {epoch}")

    for idx in pbar:
        sample_digits = val_dataset.data[idx]
        a = val_dataset._digits_to_string(sample_digits[0])
        b = val_dataset._digits_to_string(sample_digits[1])

        gt_steps = val_dataset.run_algorithm(a, b)

        current_grid = torch.ones((1, val_dataset.h, val_dataset.w), dtype=torch.long) * val_dataset.token_to_idx.index(
            "\n"
        )

        final_grid = torch.ones((1, val_dataset.h, val_dataset.w), dtype=torch.long) * val_dataset.token_to_idx.index(
            "\n"
        )

        # Parse the first step string
        s = gt_steps[0]
        t = gt_steps[-1]
        for i in range(val_dataset.h):
            for j in range(val_dataset.w - 1):  # \n already filled
                current_grid[0, i, j] = val_dataset.token_to_idx.index(s.split("\n")[i][j])
                final_grid[0, i, j] = val_dataset.token_to_idx.index(t.split("\n")[i][j])

        current_grid = trainer.fabric.to_device(current_grid)
        final_grid = trainer.fabric.to_device(final_grid)

        for _ in range(steps_to_rollout):
            with torch.no_grad():
                _, _, y_pred, _, _, _, _, _, _ = trainer.forward((current_grid, current_grid))
                out = trainer.head.step(current_grid, y_pred)
                current_grid = out

        pred_grid = current_grid[0].cpu()
        gt_grid = final_grid[0].cpu()

        mismatches = (pred_grid != gt_grid).sum().item()

        if mismatches <= 1:
            correct_count += 1

    accuracy = correct_count / n_samples
    logger.info(f"[{k}] Rollout off-by-one Accuracy: {100*accuracy:.2f}%")

    return accuracy


METRIC_REGISTRY: Dict[str, Callable] = {
    "step_accuracy": metric_step_accuracy,
    "full_sample_accuracy": metric_full_sample_accuracy,
    "off_by_one_accuracy": metric_full_sample_off_by_one_accuracy,
}


class PrintSuppressor:
    """Context manager to suppress stdout, stderr, and logging."""

    def __enter__(self):
        self._null_file = open(os.devnull, "w")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self._null_file
        sys.stderr = self._null_file
        self.logger = logging.getLogger()
        self._original_log_level = self.logger.level
        self.logger.setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.setLevel(self._original_log_level)
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._null_file.close()


def load_trainer(model_path_str: str, ckpt_name: str, n_samples: int, digits: int) -> Trainer:
    """Load trainer instance.

    Args:
        model_path_str: Path to model output directory.
        ckpt_name: Checkpoint filename.
        n_samples: Number of samples for the dataset.
        digits: Number of digits for the dataset.

    Returns:
        Trainer: Initialized trainer with loaded model.
    """
    model_dir = Path(model_path_str)

    config_file = model_dir / ".hydra" / "config.yaml"
    ckpt_path = model_dir / "checkpoints" / f"{ckpt_name}"

    if not config_file.exists():
        raise FileNotFoundError(f"Config not found at {config_file}")

    with PrintSuppressor():
        cfg = OmegaConf.load(config_file)

    cfg.resume_ckpt = str(ckpt_path)
    cfg.dataset.n_samples = n_samples
    cfg.val_split = 1.0  # Use full dataset for validation (n_samples)

    cfg.dataset.max_digits = digits
    cfg.val_min_digits = digits
    cfg.val_max_digits = digits

    with PrintSuppressor():
        trainer = Trainer(cfg, model_dir, disable_logging=True)
    trainer.model.eval()

    return trainer


def run_evaluation(config_path: str):
    cfg = OmegaConf.load(config_path)

    digit_range = cfg.digits
    n_samples = cfg.samples

    output_buffer = []

    for metric_name in cfg.metrics:
        if metric_name not in METRIC_REGISTRY:
            logger.warning(f"Metric '{metric_name}' not found in registry.")
            continue

        metric_fn = METRIC_REGISTRY[metric_name]
        logger.info(f"\n{'='*10} Evaluating: {metric_name} {'='*10}")

        results = {}  # {model_name: {digit: score}}

        for model_entry in cfg.models:
            # [name, path, ckpt]
            m_name, m_path, m_ckpt = model_entry[0], model_entry[1], str(model_entry[2])

            m_ckpt = f"ckpt_{m_ckpt}.pth"

            results[m_name] = {}

            logger.info(f"Evaluating Model: {m_name}")

            for d in digit_range:
                with PrintSuppressor():
                    trainer = load_trainer(m_path, m_ckpt, n_samples, d)
                    score = metric_fn(trainer, k=d, epoch=0)

                results[m_name][d] = score
                logger.info(f"  Digits {d}: {score:.2%}")

                del trainer

        headers = ["Model"] + [f"{d}-digits" for d in digit_range]
        table_rows = []

        for m_name in results:
            row = [m_name]
            for d in digit_range:
                val = results[m_name].get(d, 0.0)
                row.append(f"{val:.2%}")
            table_rows.append(row)

        # console table
        console_table = tabulate(table_rows, headers=headers, tablefmt="rounded_outline")
        print(f"\nResults for {metric_name}:")
        print(console_table)

        # latex table
        latex_rows = []
        for m_name in results:
            row = [m_name]
            for d in digit_range:
                val = results[m_name].get(d, 0.0)
                row.append(f"{val*100:.1f}")  # Number only for latex, usually better
            latex_rows.append(row)

        latex_headers = ["Model"] + [str(d) for d in digit_range]
        latex_table = tabulate(latex_rows, headers=latex_headers, tablefmt="latex_booktabs")

        output_buffer.append(f"%% Metric: {metric_name}")
        output_buffer.append(latex_table)
        output_buffer.append("\n")

    # Write latex tables to output file
    if cfg.outfile:
        out_path = Path(cfg.outfile)
        with open(out_path, "w") as f:
            f.write("\n".join(output_buffer))
        logger.info(f"\nSaved LaTeX tables to {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Models from YAML config")
    parser.add_argument("config", type=str, help="Path to evaluation .yaml file")
    args = parser.parse_args()

    run_evaluation(args.config)
