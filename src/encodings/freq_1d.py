import torch


class Frequency1DEncoding(torch.nn.Module):
    """
    A vanilla 1D Frequency encoding.
    """

    def __init__(self):
        super().__init__()

        # todo
