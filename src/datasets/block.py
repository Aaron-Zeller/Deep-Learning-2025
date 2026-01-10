import logging

import torch
from torch import Tensor
from src.datasets.block_util import *

logger = logging.getLogger(__name__)


class BlockDataset(torch.utils.data.Dataset):
    """Dataset for addition problems represented as step-by-step 2D blackboard grids."""

    def __init__(self, op: str, n_samples: int = 10000, max_digits: int = 3, seed: int = 42):
        """Initialize addition dataset.

        Args:
            n_samples: Number of addition problems to generate.
            max_digits: Maximum number of digits in each number.
            seed: Random seed for reproducibility.
            h: Height of block
            op: Operation determines kind of block
            w: Width of block
            seq_len: Length of sequence of computation dependent on max_digits
        """
        self.n_samples = n_samples
        self.max_digits = max_digits
        self.op = op
        self.h = self.get_height()
        self.w = self.get_width()
        self.seq_len = self.get_sequence_length()
        self.seed = seed
        self.data = self._generate_data()
        self.token_to_idx = list("0123456789+-*/?$.= \n")

        logger.info(
            f"Initialized BlockDataset({self.op}) with {n_samples} samples and {max_digits} digits => {len(self)} individual steps."
        )

    def get_height(self) -> int:
        match self.op:
            case "+":
                return 4
            case "+=":
                return 4
            case "-":
                return 5
            case "-=":
                return 4
            case "*":
                return 9
            case "/":
                return 17
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def get_width(self) -> int:
        match self.op:
            case "+":
                return self.max_digits + 2
            case "+=":
                return self.max_digits + 3
            case "-":
                return self.max_digits + 2
            case "-=":
                return self.max_digits + 3
            case "*":
                return 2 * self.max_digits + 3 + 1
            case "/":
                return 2 * self.max_digits + 5 + 1
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def get_sequence_length(self) -> int:
        match self.op:
            case "+":
                return 2 * self.max_digits + 2
            case "+=":
                return 3 * self.max_digits + 2
            case "-":
                return 8 * self.max_digits + 3
            case "*":
                return (
                    self.max_digits
                    * self.max_digits
                    * (self.max_digits * self.max_digits * (self.max_digits + 1) + 8 * self.max_digits + 10)
                )
            case "/":
                return -1
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def grid_size(self) -> tuple[int, int]:
        return self.h, self.w

    def _generate_data(self) -> Tensor:
        """Generate addition problems and their solutions.

        Note: Currently doesn't handle duplicates.

        Returns:
            (num_samples, 2) Pairs of numbers to add.
        """
        rng = torch.Generator().manual_seed(self.seed)
        a = torch.randint(0, 10**self.max_digits, (self.n_samples,), generator=rng)
        b = torch.randint(0, 10**self.max_digits, (self.n_samples,), generator=rng)

        return torch.stack([a, b], dim=-1)

    def __len__(self) -> int:
        """Get dataset length.

        Note: We index each step individually, not the entire computation.

        Returns:
            Dataset length (number of step transitions).
        """
        return len(self.data) * (self.seq_len - 1)  # Can't use last step as input

    def get_example(self) -> Tensor:
        sample = self.data[0]

        a = str(sample[0].item()).zfill(self.max_digits)
        b = str(sample[1].item()).zfill(self.max_digits)

        sample = self.run_algorithm(a, b, self.op)

        steps = torch.ones((len(sample), self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")

        for b, s in enumerate(sample):
            for i in range(self.h):
                for j in range(self.w - 1):  # \n already filled
                    steps[b, i, j] = self.token_to_idx.index(s.split("\n")[i][j])

        return steps

    def to_string(self, x: Tensor) -> str:
        out = ""
        for i in range(self.h):
            for j in range(self.w - 1):  # \n already filled
                out += self.token_to_idx[x[i, j].item()]
            out += "\n"
        return out

    @staticmethod
    def run_algorithm(a: str, b: str, op: str) -> list[str]:
        match op:
            case "+":
                return run_addition(a, b)
            case "+=":
                return run_accumulation(a, b)
            case "-":
                return run_subtraction(a, b)
            case "-=":
                return run_decrementation(a, b)
            case "*":
                return run_multiplication(a, b)
            case "/":
                raise NotImplementedError(f"Not yet implemented operator: {op}")
            case _:
                raise ValueError(f"Unknown operator: {op}")

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a dataset item as 2D blackboard grid representation.

        Converts characters to their index in the token list.

        Example sequence:
          ___     ___     _0_     _0_     10_     11_
           47      47      47      47      47      47
        +  91   +  91   +  91   +  91   +  91   +  91
        -----   -----   -----   -----   -----   -----
          ___     __8     __8     _38     _38     138

        The grid has shape (h, w) = (5, max_digits + 3 + 1), where +3 is for
        left padding and +1 is for the newline character.

        Args:
            idx: Item index.

        Returns:
            Tuple of (input_step, output_step)

            - input_step: (h, w) Input step as 2D grid.
            - output_step: (h, w) Output step as 2D grid.
        """
        sample_idx = idx // self.seq_len
        seq_idx = idx % self.seq_len

        sample = self.data[sample_idx]

        a = str(sample[0].item()).zfill(self.max_digits)
        b = str(sample[1].item()).zfill(self.max_digits)

        steps = self.run_algorithm(a, b, self.op)

        inp_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")
        out_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")

        seq_idx = seq_idx % len(steps)
        next_seq_idx = seq_idx + 1 if seq_idx + 1 < len(steps) else seq_idx
        for i in range(self.h):
            for j in range(self.w - 1):  # \n already filled
                inp_step[i, j] = self.token_to_idx.index(steps[seq_idx].split("\n")[i][j])
                out_step[i, j] = self.token_to_idx.index(steps[next_seq_idx].split("\n")[i][j])

        return inp_step, out_step
