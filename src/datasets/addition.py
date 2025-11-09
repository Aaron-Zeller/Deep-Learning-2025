import torch
from torch import Tensor

from src.interfaces import DatasetBase


class AdditionDataset(DatasetBase):
    """Dataset for addition problems represented as step-by-step 2D blackboard grids."""

    def __init__(self, num_samples: int = 10000, max_digits: int = 3, seed: int = 42):
        """Initialize addition dataset.

        Args:
            num_samples: Number of addition problems to generate.
            max_digits: Maximum number of digits in each number.
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.max_digits = max_digits
        self.seq_len = 2 * (max_digits + 1)
        self.seed = seed
        self.data = self._generate_data()
        self.token_to_idx = list("0123456789+ -_\n")

    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def _generate_data(self) -> Tensor:
        """Generate addition problems and their solutions.

        Note: Currently doesn't handle duplicates.

        Returns:
            (num_samples, 2) Pairs of numbers to add.
        """
        rng = torch.Generator().manual_seed(self.seed)
        a = torch.randint(0, 10**self.max_digits, (self.num_samples,), generator=rng)
        b = torch.randint(0, 10**self.max_digits, (self.num_samples,), generator=rng)

        return torch.stack([a, b], dim=-1)

    def __len__(self) -> int:
        """Get dataset length.

        Note: We index each step individually, not the entire computation.

        Returns:
            Dataset length (number of step transitions).
        """
        return len(self.data) * (self.seq_len - 1)  # Can't use last step as input

    @staticmethod
    def run_algorithm(a: str, b: str) -> list[str]:
        """Run addition algorithm step by step.

        Args:
            a: First summand as string
            b: Second summand as string

        Returns:
            List of string representations of each step
        """
        max_digits = len(a)

        # String Format Utilities
        blank_line = lambda x: " " * 2 + "".join(str(d) if d >= 0 else "_" for d in x[::-1])
        a_line = lambda x: " " * 3 + x
        b_line = lambda x: "+" + " " * 2 + x
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

        a = str(sample[0].item()).zfill(self.max_digits)
        b = str(sample[1].item()).zfill(self.max_digits)

        steps = AdditionDataset.run_algorithm(a, b)

        h, w = 5, self.max_digits + 3 + 1
        inp_step = torch.ones((h, w), dtype=torch.uint32) * self.token_to_idx.index("\n")
        out_step = torch.ones((h, w), dtype=torch.uint32) * self.token_to_idx.index("\n")

        for i in range(h):
            for j in range(w - 1):  # \n already filled
                inp_step[i, j] = self.token_to_idx.index(steps[seq_idx].split("\n")[i][j])
                out_step[i, j] = self.token_to_idx.index(steps[seq_idx + 1].split("\n")[i][j])

        return inp_step, out_step
