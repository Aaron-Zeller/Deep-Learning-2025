from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from einops import rearrange, einsum
from typing import Any
from train import Trainer
from omegaconf import OmegaConf
from itertools import product
import numpy as np
import sys
import argparse

from traces import InferenceTrace
from utils import pretty_print_sample, overlay_grid_text

COLOR = "plasma"


class Plotter:
    """Plotting class with different visualization methods."""

    def __init__(self, dataset: Any, trainer: Trainer, output_dir: Path):
        """Initialize plotting manager.

        Args:
            dataset: Dataset instance.
            trainer: Trainer with model to use for inference.
            output_dir: Directory to model log.
        """
        self.dataset = dataset
        self.trainer = trainer
        self.plot_dir = output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Plotting settings
        plt.rcParams["figure.dpi"] = 200

    def plot_probabilities_split(self, trace: InferenceTrace, filename: str = "split_probs.pdf"):
        """Plot split head probabilities showing a single pdf with left column
        plotting the location probabilities and right the value probabilities.

        Args:
            trace: Inference trace containing model predictions.
            filename (str, optional): Output PDF filename. Defaults to "split_probs.pdf".
        """
        pdf_path = self.plot_dir / filename

        steps = trace.completion.steps
        y_preds = trace.completion.y_preds  # [(val_logits, pos_logits)]

        n_steps = len(y_preds)
        h, w = self.dataset.h, self.dataset.w
        vocab = self.dataset.token_to_idx
        n_vocab = len(vocab)

        figsize = (12, 3 * n_steps)
        fig, axes = plt.subplots(n_steps, 2, figsize=figsize, squeeze=False)

        for step_idx, (y_pred, current_step_tensor) in enumerate(zip(y_preds, steps)):
            v_logits, loc_logits = y_pred  # (b, vocab), (b, h * w)

            # Location Prediction
            ax_loc = axes[step_idx, 0]
            loc_probs = torch.softmax(loc_logits, dim=-1)[0]
            loc_map = rearrange(loc_probs, "(h w) -> h w", h=h, w=w).cpu().numpy()
            im_loc = ax_loc.imshow(loc_map, cmap="magma", vmin=0, vmax=1)

            # Overlay grid text before prediction
            current_state = current_step_tensor[0]
            grid_str = self.dataset.to_string(current_state)
            overlay_grid_text(ax_loc, loc_map, grid_str, 0, 1)

            ax_loc.set_title(f"Step {step_idx}: Target Location", fontsize=10, fontweight="bold")
            ax_loc.set_xticks([])
            ax_loc.set_yticks([])

            # Value Prediction
            ax_val = axes[step_idx, 1]
            v_probs = torch.softmax(v_logits, dim=-1)[0].cpu().numpy()

            x_indices = range(n_vocab)
            x_labels = [str(t) for t in vocab]
            bars = ax_val.bar(x_indices, v_probs, color="skyblue", edgecolor="black")

            # Highlight maximum predicted value
            chosen_idx = v_probs.argmax()
            bars[chosen_idx].set_color("orange")
            bars[chosen_idx].set_edgecolor("black")

            ax_val.set_title(f"Step {step_idx}: Value Distribution", fontsize=10, fontweight="bold")
            ax_val.set_ylim(0, 1.1)
            ax_val.set_xticks(x_indices)
            ax_val.set_xticklabels(x_labels, fontsize=8)

            score = v_probs[chosen_idx]
            ax_val.text(
                chosen_idx, score + 0.02, f"{score:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

        plt.suptitle("Split Head Predictions: Location vs Value", fontsize=16)
        plt.tight_layout()

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved split probability summary to {pdf_path}")

    def plot_probabilities_global(self, trace: InferenceTrace, filename: str = "token_probs_grid.pdf"):
        """Plot a grid where Y-axis is Time (Steps) and X-axis is Token Class.
        Normalized over 3D grid (H, W, Vocab) at each step.
        Create a pdf with one page showing the full grid.

        Args:
            trace: Inference trace with predictions.
            filename: Filename for the output PDF. Defaults to "token_probs_grid.pdf".
        """
        pdf_path = self.plot_dir / filename

        steps = trace.completion.steps
        y_preds = trace.completion.y_preds

        tokens_to_plot = [t for t in self.dataset.token_to_idx if t in "0123456789"]

        n_steps = len(y_preds)
        n_tokens = len(tokens_to_plot)
        n_vocab = len(self.dataset.token_to_idx)
        h, w = self.dataset.h, self.dataset.w

        figsize = (n_tokens * 2, n_steps * 2)
        fig, axes = plt.subplots(n_steps, n_tokens, figsize=figsize, squeeze=False)

        for step_idx, y_pred in enumerate(y_preds):
            y_pred_reshaped = rearrange(y_pred, "b (h w) v -> b (h w v)", h=h, w=w)
            probs = torch.softmax(y_pred_reshaped, dim=-1)[0]
            probs = rearrange(probs, "(h w v) -> h w v", h=h, w=w, v=n_vocab)

            # Prepare Text Overlay
            current_state = steps[step_idx][0]
            grid_str = self.dataset.to_string(current_state)

            # Plot each token's probability heatmap
            for col_idx, token in enumerate(tokens_to_plot):
                ax = axes[step_idx, col_idx]
                ti = self.dataset.token_to_idx.index(token)
                token_prob = probs[:, :, ti].cpu()

                im = ax.imshow(token_prob, cmap=COLOR, vmin=0, vmax=1)
                overlay_grid_text(ax, token_prob, grid_str, 0, 1)

                # Formatting
                ax.set_xticks([])
                ax.set_yticks([])

                # Labels
                if step_idx == 0:
                    ax.set_title(f"Token '{token}'", fontsize=14, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"Step {step_idx}", fontsize=14, fontweight="bold")

        plt.suptitle("Probability Evolution: Rows=Time, Cols=Token Choice", fontsize=16)
        plt.tight_layout()

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved probability grid to {pdf_path}")

    def plot_attention_analysis(self, trace: InferenceTrace, filename: str = "attention_maps.pdf"):
        """Plot attention maps for each layer and head at each step for each updated
        position. Saves all steps into a multi-page PDF.

        @TODO: Might be interesting to plot when there was wrong prediction.

        Args:
            trace: Inference trace with attention maps.
            filename: Filename for the output PDF. Defaults to "attention_maps.pdf".
        """
        pdf_path = self.plot_dir / filename
        n_registers = self.trainer.head.n_registers
        n_layers = trace.completion.attentions[0].shape[0]
        n_heads = trace.completion.attentions[0].shape[2]
        n_steps = trace.n_steps
        h, w = self.dataset.h, self.dataset.w

        is_split = hasattr(self.trainer.head, "n_action_tokens")

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(n_steps - 1):  # last step doesn't have attention
                current_step = trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step)
                attentions = trace.completion.attentions[step_idx]  # (layers, b, heads, seq_len, seq_len)
                i, j = trace.completion.indices[step_idx]

                if is_split:
                    n_action_tokens = self.trainer.head.n_action_tokens
                    attentions = attentions[..., n_action_tokens:, n_action_tokens:]

                fig, axs = plt.subplots(
                    n_layers,
                    n_heads,
                    figsize=(n_heads * 2, n_layers * 2),
                    squeeze=False,
                )
                for lay, head in product(range(n_layers), range(n_heads)):
                    ax = axs[lay, head]
                    attn_map = attentions[
                        lay, 0, head, n_registers:, n_registers:
                    ]  # (seq_len, seq_len) and drop registers
                    attn_map = rearrange(attn_map, "(ha wa) (h w) -> ha wa h w", h=h, w=w, ha=h, wa=w)
                    im = ax.imshow(attn_map[i, j, :, :-1].cpu(), aspect="auto", cmap=COLOR)
                    overlay_grid_text(ax, attn_map[i, j, :, :-1].cpu(), grid_str, 0, 1, (i, j))
                    ax.set_title(f"Layer {lay+1} Head {head+1}", fontsize=10)
                    fig.colorbar(im, ax=ax)
                plt.suptitle(
                    f"Attention Maps at Step {step_idx}, Position ({i},{j})",
                    fontsize=16,
                )
                plt.tight_layout()
                # Save each figure to the PDF
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved attention maps to {pdf_path}")

    def plot_layer_activations(self, trace: InferenceTrace, filename: str = "activations.pdf"):
        """
        Plots the L2 Energy of activations to show where features are active.

        Args:
            trace: Inference trace with activations.
            filename: Filename for the output PDF. Defaults to "activations.pdf".
        """
        pdf_path = self.plot_dir / filename

        layer_names = list(trace.completion.activations[0].keys())  # assume same
        n_layers = len(layer_names)

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(trace.n_steps - 1):
                activations = trace.completion.activations[step_idx]

                fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3))
                if n_layers == 0:
                    axes = [axes]

                for i, name in enumerate(layer_names):
                    ax = axes[i]
                    act = activations[name]  # (1, C, H, W)

                    energy_map = torch.norm(act, p=2, dim=1).squeeze()  # (H, W)

                    im = ax.imshow(energy_map, cmap=COLOR)
                    ax.set_title(f"{name}\n(L2 Energy)")
                    ax.axis("off")

                plt.suptitle(f"Layer Activations at Step {step_idx}", fontsize=16)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved activations to {pdf_path}")

    def plot_conv_kernels(self, filename: str = "kernels.pdf"):
        """Plot all Conv2D kernels from the model into a multi-page PDF. Average over
        input channels for visualization.

        Args:
            filename: Filename for the output PDF. Defaults to "kernels.pdf".
        """
        pdf_path = self.plot_dir / filename

        # Automatically find all Conv2d layers
        conv_layers = [
            (name, module) for name, module in self.trainer.model.named_modules() if isinstance(module, torch.nn.Conv2d)
        ]

        with PdfPages(pdf_path) as pdf:
            for name, layer in conv_layers:
                weights = layer.weight.detach().cpu()  # (C_out, C_in, K, K)
                C_out = weights.shape[0]
                C_in = weights.shape[1]

                # Grid setup
                nrows = int(C_out**0.5)
                ncols = (C_out // nrows) + (1 if C_out % nrows != 0 else 0)
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
                fig.suptitle(f"Kernels: {name}\nShape: {weights.shape}", fontsize=10)

                # Handle single filter case
                if C_out == 1:
                    axes = [axes]
                axes_flat = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

                for i in range(C_out):
                    ax = axes_flat[i]
                    kernel = weights[i]  # (C_in, K, K)

                    # For visualization, average over input channels
                    kernel_img = kernel.mean(dim=0)

                    ax.imshow(kernel_img, cmap="gray" if C_in != 3 else None)
                    ax.axis("off")

                # Hide unused axes
                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].axis("off")

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved kernels to {pdf_path}")

    def plot_semantic_kernels(
        self,
        layer_name: str,
        embedding_layer: torch.nn.Embedding,
        filename: str = "semantic_kernels.pdf",
    ):
        """Plot cosine similarity between Conv2D kernel entries and token embeddings.
        Assumes the layer is a Conv2D layer with C_in matching embedding dim. Plots
        the best matching token for each kernel position and the similarity strength.

        Args:
            layer_name: Name of Conv2D layer to analyze.
            embedding_layer: Embedding layer containing token embeddings.
            filename: Filename for the output PDF. Defaults to "semantic_kernels.pdf".
        """
        pdf_path = self.plot_dir / filename

        target_layer = dict(self.trainer.model.named_modules())[layer_name]
        weights = target_layer.weight.detach().cpu()  # (C_out, C_in, K, K)

        n_filters, n_channels, k_h, k_w = weights.shape

        vocab_embeds = embedding_layer.weight.detach().cpu()  # (vocab_size, embedding_dim)

        assert n_channels == vocab_embeds.shape[1], "Channel mismatch"

        flat_weights = rearrange(weights, "co ci kh kw -> co (kh kw) ci")

        # Normalize for cosine similarity
        w_norm = torch.nn.functional.normalize(flat_weights, p=2, dim=-1)
        e_norm = torch.nn.functional.normalize(vocab_embeds, p=2, dim=-1)

        similarity = einsum(w_norm, e_norm, "co kw ci, v ci -> co kw v")  # (C_out, kw, vocab_size)

        # Find Best Matches
        max_vals, max_ids = similarity.max(dim=-1)

        max_vals = rearrange(max_vals, "co (kh kw) -> co kh kw", kh=k_h, kw=k_w)
        max_ids = rearrange(max_ids, "co (kh kw) -> co kh kw", kh=k_h, kw=k_w)

        # Setup plot grid
        nrows = int(n_filters**0.5)
        ncols = (n_filters // nrows) + (1 if n_filters % nrows != 0 else 0)
        figsize = (ncols * 1.5, nrows * 1.5)  # Bigger squares to fit text

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(f"Semantic Kernels: {layer_name}", fontsize=14)

        if n_filters == 1:
            axes = [axes]
        axes_flat = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

        for i in range(n_filters):
            ax = axes_flat[i]

            sim_map = max_vals[i].numpy()
            id_map = max_ids[i].numpy()

            # Plot Heatmap
            im = ax.imshow(sim_map, cmap=COLOR, vmin=0.0, vmax=1.0)

            # Overlay Text (Thanks Gemini)
            for y in range(k_h):
                for x in range(k_w):
                    token_idx = id_map[y, x]
                    # Convert index to string (safely handle if idx is out of range)
                    if token_idx < len(self.dataset.token_to_idx):
                        token_char = self.dataset.token_to_idx[token_idx]
                    else:
                        token_char = "?"

                    # White text if dark background, black if light
                    text_color = "white" if sim_map[y, x] > 0.6 else "black"
                    ax.text(
                        x,
                        y,
                        token_char,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight="bold",
                        fontsize=10,
                    )

            ax.set_title(f"Filter {i}", fontsize=8)
            ax.axis("off")

        # Hide unused axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.tight_layout()
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved semantic kernels to {pdf_path}")


def main(args):
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found at {model_dir}")
        sys.exit(1)

    config_file = model_dir / ".hydra" / "config.yaml"
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)

    cfg = OmegaConf.load(config_file)
    cfg.resume_ckpt = model_dir / "checkpoints" / args.ckpt

    print(f"Loading model from: {cfg.resume_ckpt}")
    trainer = Trainer(cfg, model_dir)
    trainer.model.eval()

    dataset = trainer.val_dataset
    plotter = Plotter(dataset, trainer, model_dir)

    sample = trainer.fabric.to_device(dataset.get_example())
    pretty_print_sample(dataset, sample)

    # If no activation layers are provided, we pass None/Empty to InferenceTrace
    target_layers = args.activation_layers if args.activation_layers else []

    with torch.no_grad():
        # Initialize trace with the layers we want to monitor
        trace = InferenceTrace(trainer, sample, target_layers)

    if args.plot_probabilities:
        print("Plotting Probabilities...")
        if hasattr(trainer.head, "n_action_tokens"):
            plotter.plot_probabilities_split(trace)
        else:
            plotter.plot_probabilities_global(trace)

    if args.plot_attention:
        print("Plotting Attention...")
        plotter.plot_attention_analysis(trace)

    if args.activation_layers:
        print(f"Plotting Activations for: {args.activation_layers}")
        plotter.plot_layer_activations(trace)

    if args.plot_conv_kernels:
        print("Plotting Convolution Kernels...")
        plotter.plot_conv_kernels()

    if args.semantic_kernel_layer:
        print(f"Plotting Semantic Kernels for: {args.semantic_kernel_layer}")
        plotter.plot_semantic_kernels(args.semantic_kernel_layer, trainer.model.src_embed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interpretation plots for the Transformer.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to model directory (e.g., 'outputs/name/day/time')",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt_0120.pth",
        help="Checkpoint filename (default: ckpt_0120.pth)",
    )

    # Boolean Flags
    parser.add_argument(
        "--no-probs",
        action="store_false",
        dest="plot_probabilities",
        help="Disable probability plotting",
    )
    parser.add_argument(
        "--no-attn",
        action="store_false",
        dest="plot_attention",
        help="Disable attention analysis plotting",
    )
    parser.add_argument(
        "--no-kernels",
        action="store_false",
        dest="plot_conv_kernels",
        help="Disable convolution kernel plotting",
    )

    # Set defaults to True
    parser.set_defaults(plot_probabilities=True, plot_attention=True, plot_conv_kernels=True)

    parser.add_argument(
        "--activation_layers",
        nargs="+",
        default=[],
        help="List of layer names to plot activations for (e.g. '_forward_module.lens.0' 'lens.3'). Returns empty list if not provided.",
    )

    parser.add_argument(
        "--semantic_kernel_layer",
        type=str,
        default="",
        help="Layer name for semantic kernel analysis. Leave empty to skip.",
    )

    args = parser.parse_args()

    main(args)
