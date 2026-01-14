import logging

import torch
from torch import Tensor

from src.interfaces import DatasetBase

logger = logging.getLogger(__name__)


class AdditionDataset(DatasetBase):
    """Dataset for addition problems represented as step-by-step 2D blackboard grids."""

    def __init__(self, n_samples: int = 10000, max_digits: int = 3, seed: int = 42, separate_carry: bool = False):
        """Initialize addition dataset.

        Args:
            n_samples: Number of addition problems to generate.
            max_digits: Maximum number of digits in each number.
            seed: Random seed for reproducibility.
            separate_carry: Whether to use separate carry symbols (0 -> O, 1 -> I).
        """
        self.n_samples = n_samples
        self.max_digits = max_digits
        self.h = 5
        self.w = max_digits + 3 + 1
        self.seq_len = 2 * (max_digits + 1)
        self.seed = seed
        self.data = self._generate_data()
        self.separate_carry = separate_carry
        self.token_to_idx = list("0123456789+ -_\n") if not self.separate_carry else list("0123456789+ -_OI\n")

        logger.info(
            f"Initialized AdditionDataset with {n_samples} samples and {max_digits} digits => {len(self)} individual steps. Separate carry: {separate_carry}"
        )

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def grid_size(self) -> tuple[int, int]:
        return self.h, self.w

    def _generate_data(self) -> Tensor:
        """Generate addition problems and their solutions.

        Note: Currently doesn't handle duplicates.

        Returns:
            (num_samples, 2, max_digits) Digit representation of pairs of numbers to add.
            Each element is a digit from 0-9.
        """
        rng = torch.Generator().manual_seed(self.seed)
        # Generate random digits directly: shape (n_samples, 2, max_digits)
        digits = torch.randint(0, 10, (self.n_samples, 2, self.max_digits), generator=rng)
        return digits

    def __len__(self) -> int:
        """Get dataset length.

        Note: We index each step individually, not the entire computation.

        Returns:
            Dataset length (number of step transitions).
        """
        return len(self.data) * (self.seq_len - 1)  # Can't use last step as input

    def _digits_to_string(self, digits: Tensor) -> str:
        """Convert digit array to string.

        Args:
            digits: (max_digits,) tensor of digits

        Returns:
            String representation of the number
        """
        return "".join(str(d.item()) for d in digits)

    def get_example(self) -> Tensor:
        sample = self.data[0]

        a = self._digits_to_string(sample[0])
        b = self._digits_to_string(sample[1])

        sample = self.run_algorithm(a, b, self.separate_carry)

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
    def run_algorithm(a: str, b: str, separate_carry: bool = False) -> list[str]:
        """Run addition algorithm step by step.

        Args:
            a: First summand as string
            b: Second summand as string
            separate_carry: Whether to use separate carry symbols

        Returns:
            List of string representations of each step
        """
        max_digits = len(a)

        # String Format Utilities
        blank_line = lambda x: " " * 2 + "".join(str(d) if d >= 0 else "_" for d in x[::-1])
        carry_line = (
            blank_line
            if not separate_carry
            else lambda x: " " * 2 + "".join(["O", "I"][d] if d >= 0 else "_" for d in x[::-1])
        )
        a_line = lambda x: " " * 3 + x
        b_line = lambda x: "+" + " " * 2 + x
        sep_line = lambda: "-" * (max_digits + 3)
        make_step = lambda _carry, _a, _b, _out: "\n".join(
            [carry_line(_carry), a_line(_a), b_line(_b), sep_line(), blank_line(_out)]
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
            s = da + db + c

            out[i] = s % 10
            steps.append(make_step(carry, a, b, out))

            carry[i + 1] = s // 10
            steps.append(make_step(carry, a, b, out))

        # Final step
        out[-1] = carry[-1]
        steps.append(make_step(carry, a, b, out))

        return steps

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

        a = self._digits_to_string(sample[0])
        b = self._digits_to_string(sample[1])

        steps = AdditionDataset.run_algorithm(a, b, self.separate_carry)

        inp_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")
        out_step = torch.ones((self.h, self.w), dtype=torch.long) * self.token_to_idx.index("\n")

        next_seq_idx = seq_idx + 1 if seq_idx + 1 < len(steps) else seq_idx
        for i in range(self.h):
            for j in range(self.w - 1):  # \n already filled
                inp_step[i, j] = self.token_to_idx.index(steps[seq_idx].split("\n")[i][j])
                out_step[i, j] = self.token_to_idx.index(steps[next_seq_idx].split("\n")[i][j])

        return inp_step, out_step
