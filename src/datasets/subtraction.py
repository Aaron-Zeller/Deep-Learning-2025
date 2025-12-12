import logging

import torch
from torch import Tensor

from src.interfaces import DatasetBase

logger = logging.getLogger(__name__)


class SubtractionDataset(DatasetBase):
    """Dataset for subtraction problems represented as step-by-step 2D blackboard grids."""

    def __init__(self, n_samples: int = 10000, max_digits: int = 3, seed: int = 42):
        """Initialize subtraction dataset.

        Args:
            n_samples: Number of subtraction problems to generate.
            max_digits: Maximum number of digits in each number.
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.max_digits = max_digits
        self.h = 5
        self.w = max_digits + 3 + 1
        self.seq_len = 2 * (max_digits + 1)
        self.seed = seed
        self.data = self._generate_data()
        self.token_to_idx = list("0123456789+ -_*\n")

        logger.info(
            f"Initialized SubtractionDataset with {n_samples} samples and {max_digits} digits => {len(self)} individual steps."
        )

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def grid_size(self) -> tuple[int, int]:
        return self.h, self.w

    def _generate_data(self) -> Tensor:
        """Generate subtraction problems and their solutions.

        Note: Currently doesn't handle duplicates.

        Returns:
            (num_samples, 2) Pairs of numbers to subtract.
        """
        rng = torch.Generator().manual_seed(self.seed)
        a = torch.randint(0, 10**self.max_digits, (self.n_samples,), generator=rng)
        b = torch.randint(0, 10**self.max_digits, (self.n_samples,), generator=rng)

        # it needs to hold that a >= b, otherwise we get negative numbers
        mask = a < b
        larger = a
        smaller = b

        larger[mask] = b[mask]
        smaller[mask] = a[mask]

        return torch.stack([larger, smaller], dim=-1)

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

        sample = self.run_algorithm(a, b)

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
    def run_algorithm(a: str, b: str) -> list[str]:
        """Run subtraction algorithm step by step.

        Args:
            a: First number as string
            b: Second number which is subtracted from the first one as string

        Returns:
            List of string representations of each step
        """
        max_digits = len(a)

        # String Format Utilities
        blank_line = lambda x: " " * 2 + "".join(str(d) if d >= 0 else "_" for d in x[::-1])
        a_line = lambda x: " " * 3 + x
        b_line = lambda x: "-" + " " * 2 + x
        sep_line = lambda: "-" * (max_digits + 3)
        make_step = lambda _carry, _a, _b, _out: "\n".join(
            [blank_line(_carry), a_line(_a), b_line(_b), sep_line(), blank_line(_out)]
        )

        # Algorithm State
        carry = [-1 for _ in range(max_digits + 1)]
        out = [-1 for _ in range(max_digits + 1)]

        # Initial step
        steps = [make_step(carry, a, b, out)]

        for i in range(max_digits):  # Each iteration produces two steps (compute digit + carry)
            c = 0 if i == 0 else carry[i]
            da = int(a[-i - 1])
            db = int(b[-i - 1])
            s = da - db - c

            out[i] = s % 10
            steps.append(make_step(carry, a, b, out))

            if s < 0:
                carry[i + 1] = 1
            else:
                carry[i + 1] = 0
            steps.append(make_step(carry, a, b, out))

        # Final step
        out[-1] = carry[-1]
        steps.append(make_step(carry, a, b, out))

        return steps

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a dataset item as 2D blackboard grid representation.

        Converts characters to their index in the token list.

        Example sequence:
          ___     ___     _1_     _1_     01_     01_
           91      91      91      91      91      91
        -  47   -  47   -  47   -  47   -  47   -  47
        -----   -----   -----   -----   -----   -----
          ___     __4     __4     _44     _44     044

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

        steps = SubtractionDataset.run_algorithm(a, b)

        inp_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")
        out_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")

        next_seq_idx = seq_idx + 1 if seq_idx + 1 < len(steps) else seq_idx
        for i in range(self.h):
            for j in range(self.w - 1):  # \n already filled
                inp_step[i, j] = self.token_to_idx.index(steps[seq_idx].split("\n")[i][j])
                out_step[i, j] = self.token_to_idx.index(steps[next_seq_idx].split("\n")[i][j])

        return inp_step, out_step
