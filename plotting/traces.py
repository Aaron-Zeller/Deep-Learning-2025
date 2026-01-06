from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import torch
from torch import Tensor
from train import Trainer
from utils import ActivationMonitor


@dataclass
class TraceHistory:
    """Container for the history of a specific inference mode."""

    steps: List[Tensor] = field(default_factory=list)  # Calculation steps
    indices: List[Tuple[int, int]] = field(default_factory=list)  # Indices of changed elements
    attentions: List[Tensor] = field(default_factory=list)  # Attention weights
    y_preds: List[Tensor] = field(default_factory=list)  # Predicted distributions
    activations: List[Dict[str, Tensor]] = field(default_factory=list)  # Recorded activations


class InferenceTrace:
    """Runs inference on a sample and records traces for:
    - Single-step prediction using previous GT step
    - Completion prediction using previous predicted step

    This could be used to analyze error accumulation over steps and recovery?
    """

    def __init__(self, trainer: Trainer, sample: Tensor, layer_names: List[str] = []):
        """Initialize inference trace on sample.

        Args:
            trainer: Trainer with model to use for inference.
            sample: Sample calculation steps
            layer_names: Names of layers to monitor. Defaults to [].
        """

        self.gt = TraceHistory()  # ground truth steps
        self.single = TraceHistory()  # using previous GT step
        self.completion = TraceHistory()  # using previous pred step

        self.n_steps = sample.shape[0]
        self.error_steps: List[int] = []
        self.layer_names = layer_names

        self._run_inference(trainer, sample)

    def _run_inference(self, trainer: Trainer, sample: Tensor) -> None:
        """Runs the inference process and records traces.

        Args:
            trainer: Trainer with model to use for inference.
            sample: Sample calculation steps
        """
        # Initialize with the first step (t=0)
        initial_step = sample[0:1]
        self.gt.steps.append(initial_step)
        self.single.steps.append(initial_step)
        self.completion.steps.append(initial_step)

        # Initialize the monitor for activations if needed
        with ActivationMonitor(trainer.model, self.layer_names) as monitor:
            for i in range(1, self.n_steps):
                self.gt.steps.append(sample[i : i + 1])
                self._record_diff(self.gt, i, sample[i - 1 : i])

                # Single step inference using previous GT
                self._predict_step(
                    trainer,
                    input_step=sample[i - 1 : i],
                    history=self.single,
                    monitor=monitor,
                )

                # Completion inference using previous predicted step
                self._predict_step(
                    trainer,
                    input_step=self.completion.steps[-1],
                    history=self.completion,
                    monitor=monitor,
                )

    def _predict_step(
        self,
        trainer: Trainer,
        input_step: Tensor,
        history: TraceHistory,
        monitor: ActivationMonitor,
    ) -> None:
        """Helper method to run inference on single step with monitors.

        Args:
            trainer: Trainer with model to use for inference.
            input_step: Input step for inference.
            history: History object to update with results.
            monitor: Activation monitor to track layer activations.
        """

        # Clear monitor activations, as these don't get reset automatically :)
        monitor.activations = {}

        outputs = trainer.forward((input_step, None))

        y_pred = outputs[2]
        enc_attns = outputs[5]

        next_step = trainer.head.step(input_step, y_pred)

        # Update History
        history.steps.append(next_step)
        history.y_preds.append(y_pred)
        history.attentions.append(enc_attns)
        history.activations.append(monitor.activations.copy())

        # Record diff indices
        diff_indices = (next_step[0] != input_step[0]).nonzero(as_tuple=False)
        history.indices.append(tuple(diff_indices.tolist()[0]))

    def _record_diff(self, history: TraceHistory, current_idx: int, prev_step: Tensor) -> None:
        """Calculates which grid cell has been updated and stores the index.

        Args:
            history: History object to update with new index.
            current_idx: Current step index.
            prev_step: Previous step tensor.
        """
        current_step = history.steps[current_idx]
        diff = (current_step[0] != prev_step[0]).nonzero(as_tuple=False)
        history.indices.append(tuple(diff.tolist()[0]))

    def _check_errors(self) -> None:
        """Calculates all the indices, where the full sequence inference has made errros."""
        for i in range(self.n_steps):
            if not torch.equal(self.completion.steps[i], self.gt.steps[i]):
                self.error_steps.append(i)
