import os
import sys
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Callable
from tabulate import tabulate
import logging
import torch
from omegaconf import OmegaConf

# Assuming 'train' is available in your python path
from train import Trainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# To add a new metric, define a function here and add it to the METRIC_REGISTRY.
# Signature: (trainer, k_idx, epoch, unrolled_grid) -> float
def metric_step_accuracy(trainer: Trainer, k: int, epoch: int = 0, unrolled_grid=None) -> float:
    """Wrapper for the existing per-step validation."""
    if hasattr(trainer, "val_k_epoch"):
        _, acc = trainer.val_k_epoch(k, 0)
        return acc
    _, acc = trainer.val_epoch(0)
    return acc


def metric_full_sample_accuracy(trainer: Trainer, k: int, epoch: int = 0, unrolled_grid=None) -> float:
    """Exact Match on unrolled grid."""
    if not unrolled_grid:
        return 0.0

    correct = 0
    total = len(unrolled_grid)

    for pred_grid, gt_grid in unrolled_grid:
        if torch.equal(pred_grid, gt_grid):
            correct += 1

    return correct / total


def metric_full_sample_off_by_one(trainer: Trainer, k: int, epoch: int = 0, unrolled_grid=None) -> float:
    """Full step accuracy allowing off-by-one errors."""
    if not unrolled_grid:
        return 0.0

    pass_count = 0
    total = len(unrolled_grid)

    for pred_grid, gt_grid in unrolled_grid:
        if (torch.abs(pred_grid - gt_grid) <= 1).all():
            pass_count += 1

    return pass_count / total


METRIC_REGISTRY: Dict[str, Callable] = {
    "step_accuracy": metric_step_accuracy,
    "full_sample_accuracy": metric_full_sample_accuracy,
    "full_sample_off_by_one": metric_full_sample_off_by_one,
}
ROLLOUT_METRICS = {"full_sample_accuracy", "full_sample_off_by_one"}


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


def run_evaluation(config_path: str, args: argparse.Namespace):
    cfg = OmegaConf.load(config_path)

    if args.model:
        # args.model is a list of lists: [['Name', 'Path', 'Ckpt'], ...]
        cfg.models = args.model
        logger.info(f"Overriding models list with {len(args.model)} model(s) from CLI.")

    if args.samples:
        cfg.samples = args.samples
        logger.info(f"Overriding samples with: {cfg.samples}")

    # Determine digit range based on CLI args or config defaults
    # Priority: CLI Args > Config Range Values > Default Fallback
    c_min = args.min_digit if args.min_digit is not None else cfg.get("min_digit", 1)
    c_max = args.max_digit if args.max_digit is not None else cfg.get("max_digit", 100)
    c_stride = args.stride if args.stride is not None else 1
    c_start = args.start_digit if args.start_digit is not None else c_min

    # Check if we should generate a range or use the explicit list from config
    use_range_generation = (
        args.min_digit is not None
        or args.max_digit is not None
        or args.stride is not None
        or args.start_digit is not None
    )

    if use_range_generation:
        cfg.digits = list(range(c_start, c_max + 1, c_stride))
        logger.info(f"Generated digit range: {cfg.digits} (start={c_start}, max={c_max}, stride={c_stride})")
    elif not hasattr(cfg, "digits") or not cfg.digits:
        # Fallback if no specific list exists in YAML and no CLI range provided
        cfg.digits = list(range(c_min, c_max + 1, c_stride))
        logger.info(f"Using default digit range: {cfg.digits}")

    output_dir = Path(args.output_dir) if args.output_dir else (Path(cfg.outfile).parent if cfg.outfile else Path("."))
    output_dir.mkdir(parents=True, exist_ok=True)

    digit_range = cfg.digits
    n_samples = cfg.samples

    output_buffer = []

    # [metric_name][model_name][digit] = score
    all_results = {m: {} for m in cfg.metrics if m in METRIC_REGISTRY}

    for model_entry in cfg.models:
        m_name, m_path, m_ckpt = model_entry[0], model_entry[1], str(model_entry[2])
        m_ckpt = f"ckpt_{m_ckpt}.pth"

        # Initialize this model in the results dict for all metrics
        for m in all_results:
            all_results[m][m_name] = {}

        logger.info(f"Evaluating Model: {m_name}")
        for d in digit_range:
            with PrintSuppressor():
                trainer = load_trainer(m_path, m_ckpt, n_samples, d)

                unrolled_grid = None
                if any(m in ROLLOUT_METRICS for m in cfg.metrics):
                    if hasattr(trainer, "unroll_evaluation"):
                        unrolled_grid = trainer.unroll_evaluation(k=d)

                for metric_name in cfg.metrics:
                    if metric_name not in METRIC_REGISTRY:
                        continue

                    metric_fn = METRIC_REGISTRY[metric_name]
                    score = metric_fn(trainer, k=d, epoch=0, unrolled_grid=unrolled_grid)

                    all_results[metric_name][m_name][d] = score

            log_parts = [f"{m}={all_results[m][m_name][d]:.2%}" for m in cfg.metrics if m in all_results]
            logger.info(f"  Digits {d}: " + ", ".join(log_parts))

            del trainer
            torch.cuda.empty_cache()

    # CSV export per digit
    for d in digit_range:
        csv_path = output_dir / f"eval_results_digit_{d}.csv"
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["Model"] + list(cfg.metrics)
                writer.writerow(header)

                for model_entry in cfg.models:
                    m_name = model_entry[0]
                    row = [m_name]
                    for metric in cfg.metrics:
                        if metric in all_results and m_name in all_results[metric]:
                            score = all_results[metric][m_name].get(d, None)
                            row.append(f"{score:.6f}" if score is not None else "")
                        else:
                            row.append("")
                    writer.writerow(row)
            logger.info(f"Saved CSV results for {d} digits to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to write CSV for digit {d}: {e}")

    # Print and store latex tables
    for metric_name in cfg.metrics:
        if metric_name not in all_results:
            continue

        logger.info(f"\n{'='*10} Evaluating: {metric_name} {'='*10}")
        results = all_results[metric_name]

        headers = ["Model"] + [f"{d}-digits" for d in digit_range]

        # Prepare Rows
        table_rows = []
        latex_rows = []

        for m_name in results:
            console_row = [m_name]
            latex_row = [m_name]
            for d in digit_range:
                val = results[m_name].get(d, 0.0)
                console_row.append(f"{val:.2%}")
                latex_row.append(f"{val*100:.1f}")
            table_rows.append(console_row)
            latex_rows.append(latex_row)

        # Print Console Table
        print(f"\nResults for {metric_name}:")
        print(tabulate(table_rows, headers=headers, tablefmt="rounded_outline"))

        # Store LaTeX Table
        latex_headers = ["Model"] + [str(d) for d in digit_range]
        latex_table = tabulate(latex_rows, headers=latex_headers, tablefmt="latex_booktabs")

        output_buffer.append(f"%% Metric: {metric_name}")
        output_buffer.append(latex_table)
        output_buffer.append("\n")

    if cfg.outfile:
        out_path = Path(cfg.outfile)
        with open(out_path, "w") as f:
            f.write("\n".join(output_buffer))
        logger.info(f"\nSaved LaTeX tables to {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Models from YAML config")
    parser.add_argument("config", type=str, help="Path to evaluation .yaml file")

    # Overrides
    parser.add_argument("--samples", type=int, help="Override number of samples")
    parser.add_argument("--output-dir", type=str, help="Directory to save CSVs")

    # Range / Batching Arguments
    parser.add_argument("--min-digit", type=int, help="Minimum digit length (defaults to config min)")
    parser.add_argument("--max-digit", type=int, help="Maximum digit length (defaults to config max)")
    parser.add_argument("--stride", type=int, help="Step size for digit range (default 1)")
    parser.add_argument("--start-digit", type=int, help="Specific start point for this run (defaults to min)")

    # Model override: Can be used multiple times
    # Usage: --model "Name" "Path" "Checkpoint"
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "PATH", "CKPT"),
        help="Override models list. Usage: --model 'Best' 'path/to/run' '100'",
    )

    args = parser.parse_args()

    run_evaluation(args.config, args)
