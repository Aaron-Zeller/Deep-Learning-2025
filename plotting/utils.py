from typing import Any, List, Callable, Self, Optional
from torch import Tensor


def pretty_print_sample(dataset: Any, x: Tensor, verbose: bool = True) -> str:
    """Pretty print a data sample.

    Args:
        dataset: Dataset object to use for string conversion.
        x: Tensor representing the sample.
        verbose: Whether to print the output. Defaults to True.

    Returns:
        str: Formatted string representation of the sample.
    """
    steps = [dataset.to_string(x[i]).split("\n") for i in range(x.shape[0])]

    x_str = "\n".join("   ".join(steps[i][j] for i in range(len(steps))) for j in range(len(steps[0])))

    if verbose:
        print(x_str)

    return x_str


class ActivationMonitor:
    """Context manager to record intermediate activations from specified layers."""

    def __init__(self, model, layer_names: List[str]):
        """Initialize activation monitor. Layer names can be found via model.named_modules().

        Args:
            model: Model to monitor.
            layer_names: Names of layers to monitor.
        """
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []  # to store pytorch hooks

    def _get_hook(self, name: str) -> Callable:
        """Get hook capture method.

        Args:
            name: Name of the layer to capture activations from.

        Returns:
            Callable: Hook function to capture activations.
        """

        def hook(model, input, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def __enter__(self) -> Self:
        """Connect hooks to the model layers.

        Returns:
            ActivationMonitor: The activation monitor instance.
        """
        # Register hooks on layers that match the given names
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Clean up hooks so they don't slow down future inference
        for hook in self.hooks:
            hook.remove()


def overlay_grid_text(
    ax,
    data: Tensor,
    grid_str: str,
    vmin: float,
    vmax: float,
    idx: Optional[int] = None,
) -> None:
    """Helper to overlay character text onto heatmaps."""
    h, w = data.shape
    vhalf = (vmin + vmax) / 2
    lines = grid_str.split("\n")

    for y in range(h):
        for x in range(w):
            if x < len(lines[y]):
                char = lines[y][x]

                if idx is not None:
                    (i, j) = idx
                    if (y, x) == (i, j):
                        char = "X"

                color = "white" if data[y, x] < vhalf else "black"
                ax.text(x, y, char, ha="center", va="center", color=color, fontsize=6)
