import logging

import torch
from torch import Tensor
import numpy as np

from src.interfaces import DatasetBase

logger = logging.getLogger(__name__)


class MultiplicationDataset(DatasetBase):
    """Dataset for multiplication problems represented as step-by-step 2D blackboard grids."""

    def __init__(self, n_samples: int = 10000, max_digits: int = 2, seed: int = 42):
        """Initialize multiplication dataset.

        Args:
            n_samples: Number of multiplication problems to generate.
            max_digits: Maximum number of digits in each number.
            seed: Random seed for reproducibility.
        """
        self.n_samples = n_samples
        self.max_digits = max_digits
        self.h = 5 + max_digits**2
        self.w = max_digits * 2 + 3 + 1

        self.seq_len = get_seq_len(max_digits)
        self.seed = seed
        self.data = self._generate_data()
        self.token_to_idx = list("0123456789+ -_*\n")

        logger.info(
            f"Initialized MultiplicationDataset with {n_samples} samples and {max_digits} digits => {len(self)} individual steps."
        )

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def grid_size(self) -> tuple[int, int]:
        return self.h, self.w

    def _generate_data(self) -> Tensor:
        """Generate multiplication problems and their solutions.

        Note: Currently doesn't handle duplicates.

        Returns:
            (num_samples, 2) Pairs of numbers to multiply.
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
    def run_algorithm(a: str, b: str, max_steps: int = None) -> list[str]:
        """Run multiplication algorithm step by step.

        Args:
            a: First number as string
            b: Second number as string
            max_steps: Upperbound the maximal number of steps to be computed. Compute everything if None

        Returns:
            List of string representations of each step
        """
        max_digits = len(a)

        max_width = max_digits * 2  # maximum width of one mult
        n_mults = max_digits**2  # number of mults to be added up

        # String Format Utilities
        blank_line = lambda x: " " * 3 + "".join(str(d) if d >= 0 else "_" for d in x[-2::-1])
        a_line = lambda x: " " * (3 + max_width - len(x)) + x
        b_line = lambda x: " " * (max_width - len(x)) + "*" + " " * 2 + x
        mult_line = lambda x, row: " " * 3 + "".join(str(d) if d >= 0 else " " for d in x[row])
        mult_lines = lambda x: [mult_line(x, row) for row in range(n_mults)]
        sep_line = lambda: " " * (max_digits) + "-" * (max_digits + 3)
        make_step = lambda _carry, _a, _b, mults, _out,: "\n".join(
            [a_line(_a), b_line(_b), sep_line(), *mult_lines(mults), blank_line(_carry), blank_line(_out)]
        )

        # Algorithm State
        carry = [-1 for _ in range(max_width + 1)]
        mults = [[-1 for _ in range(max_width)] for _ in range(n_mults)]
        out = [-1 for _ in range(max_width + 1)]

        # Do the mults first:

        # Initial step
        steps = [make_step(carry, a, b, mults, out)]

        for da in range(max_digits):
            for db in range(max_digits):
                row = da * max_digits + db
                num_zeros = da + db
                for i in range(num_zeros):  # First add the zeros on the right (if there are any)
                    mults[row][-i - 1] = 0
                    steps.append(make_step(carry, a, b, mults, out))
                # Then do the actual mults, at most 2 digits
                ai = int(a[-da - 1])
                bi = int(b[-db - 1])
                mult = ai * bi

                mults[row][-num_zeros - 1] = mult % 10
                steps.append(make_step(carry, a, b, mults, out))

                mults[row][-num_zeros - 2] = mult // 10
                steps.append(make_step(carry, a, b, mults, out))

                if max_steps is not None and len(steps) >= max_steps:
                    return steps

        # Add everything up

        for i in range(max_width):  # Each iteration produces two steps (compute digit + carry)
            c = 0 if i == 0 else carry[i]
            s = sum([x for x in np.array(mults)[:, -i - 1] if x >= 0])
            s += c

            out[i] = s % 10
            steps.append(make_step(carry, a, b, mults, out))

            carry[i + 1] = s // 10
            steps.append(make_step(carry, a, b, mults, out))
        # there is never a carry at the end
        steps.pop()

        return steps

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a dataset item as 2D blackboard grid representation.

        Converts characters to their index in the token list.

        Example final state:

           99
        *  99
        -----
           81
          810
          810
         8100
         1100
         9801

        Args:
            idx: Item index.

        Returns:
            Tuple of (input_step, output_step)

            - input_step: (h, w) Input step as 2D grid.
            - output_step: (h, w) Output step as 2D grid.
        """
        sample_idx = idx // self.seq_len
        seq_idx = idx % self.seq_len
        # TODO: seq_idx should be chosen differently, as for large max_digits the multiplication of two small numbers will mostly return no-ops (as in just pasting 0)
        sample = self.data[sample_idx]

        a = str(sample[0].item()).zfill(self.max_digits)
        b = str(sample[1].item()).zfill(self.max_digits)

        steps = MultiplicationDataset.run_algorithm(a, b, seq_idx + 1)

        inp_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")
        out_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")

        next_seq_idx = seq_idx + 1 if seq_idx + 1 < len(steps) else seq_idx
        for i in range(self.h):
            for j in range(self.w - 1):  # \n already filled
                inp_step[i, j] = self.token_to_idx.index(steps[seq_idx].split("\n")[i][j])
                out_step[i, j] = self.token_to_idx.index(steps[next_seq_idx].split("\n")[i][j])

        return inp_step, out_step


def get_seq_len(max_digits: int) -> int:
    """Calculates the seq_len for max_digits digits in the multiplication dataset
    Args:
        max_digits: Maximum number of digits.
    Returns:
        Int representing the seq_len.
    """
    # First get the number of digits produced by multiplications:
    # sum (sum (a+b)) = sum (a*max_digits + (max_digits * (max_digits+1)/2))
    #             = max_digits * (sum (a) + max_digits * (max_digits+1)/2)
    #             = max_digits * (max_digits * (max_digits+1))
    seq_len = max_digits * (max_digits * (max_digits + 1))
    # Number of digits to write at the bottom resulting from addition
    seq_len += max_digits * 2
    # All of them have a carry step
    seq_len += max_digits * 2
    return seq_len
