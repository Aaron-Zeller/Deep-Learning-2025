from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hydra_zen import instantiate
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from einops import rearrange, einsum
from typing import Any
from train import Trainer
from omegaconf import OmegaConf
from itertools import product
import numpy as np
import sys
import argparse
import tempfile
import subprocess
import shutil

from traces import InferenceTrace
from utils import pretty_print_sample, overlay_grid_text
import matplotlib as mpl

from matplotlib import rcParams

rcParams["font.family"] = "Times New Roman"

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
                chosen_idx,
                score + 0.02,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
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

    def _compute_averaged_energy_maps(self, traces: list[InferenceTrace]):
        """
        Computes L2 energy maps averaged across traces for each step and layer.

        Returns:
            avg_energy_data: List of length n_steps. Each element is a dict {layer_name: Tensor(H, W)}.
            layer_names: Ordered list of layer names.
            n_steps: Number of steps processed.
        """
        if not traces or not traces[0].completion.activations:
            print("No traces or activations found")
            return None, None, 0

        first_trace = traces[0]
        layer_names = list(first_trace.completion.activations[0].keys())
        n_steps = max(0, first_trace.n_steps - 1)

        avg_energy_data = []

        for step_idx in range(n_steps):
            step_maps = {}
            for name in layer_names:
                per_trace_maps = []
                for tr in traces:
                    act = tr.completion.activations[step_idx][name]  # (1, C, H, W)
                    # Compute L2 norm -> (H, W)
                    energy = torch.norm(act, p=2, dim=1).squeeze().cpu()
                    per_trace_maps.append(energy)

                # Stack and average -> (H, W)
                stacked = torch.stack(per_trace_maps, dim=0)
                step_maps[name] = stacked.mean(dim=0)

            avg_energy_data.append(step_maps)

        return avg_energy_data, layer_names, n_steps

    def plot_layer_activations_merged(
        self,
        traces: list[InferenceTrace],
        filename: str = "activations_merged.pdf",
        step_id: int | None = None,
        layer_names_to_plot: list[str] | None = None,
    ):
        """Plot L2 energy over channels over multiple steps averaged over multiple
        traces and concatenated column wise.

        If step_id and layer_names_to_plot provided, highlights that step with a border.

        Args:
            traces: Inference traces
            filename: Output PDF filename
            step_id: Specific step to highlight (optional)
            layer_names_to_plot: Specific layers to highlight (optional)
        """
        pdf_path = self.plot_dir / filename

        avg_energy_data, all_layer_names, n_steps = self._compute_averaged_energy_maps(traces)
        if not avg_energy_data:
            return

        sample_map = avg_energy_data[0][all_layer_names[0]]
        H, W = sample_map.shape

        vmax_per_layer = {name: 0.0 for name in all_layer_names}
        for step_maps in avg_energy_data:
            for name, energy_map in step_maps.items():
                vmax_per_layer[name] = max(vmax_per_layer[name], float(energy_map.max().item()))

        concat_cols = []
        for name in all_layer_names:
            layer_steps = [avg_energy_data[s][name] for s in range(n_steps)]
            col = torch.cat(layer_steps, dim=0)
            concat_cols.append(col)

        full_concat = torch.cat(concat_cols, dim=1)

        global_vmax = max(max(vmax_per_layer.values()), 1e-8)
        vmin, vmax = 0.0, global_vmax

        per_col_w = 0.6
        per_step_h = 0.06
        fig_w = max(1.2, len(all_layer_names) * per_col_w)
        fig_h = max(1.2, n_steps * per_step_h)

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        plt.subplots_adjust(wspace=0.0, hspace=0.00)

        with PdfPages(pdf_path) as pdf:
            im = ax.imshow(
                full_concat.numpy(),
                cmap=COLOR,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
            )

            # Draw separators
            for s in range(1, n_steps):
                y = s * H - 0.5
                ax.axhline(y=y, color="white", linewidth=0.4, alpha=0.6)
            for c in range(1, len(all_layer_names)):
                x = c * W - 0.5
                ax.axvline(x=x, color="white", linewidth=0.4, alpha=0.6)

            # Determine which rows steps and layers to highlight
            target_steps = [step_id] if step_id is not None else range(n_steps)

            if layer_names_to_plot is not None:
                # Find indices of requested layers
                target_layer_indices = [i for i, name in enumerate(all_layer_names) if name in layer_names_to_plot]
            else:
                target_layer_indices = range(len(all_layer_names))

            # Only draw borders if we are restricting view (i.e. arguments are not None)
            if step_id is not None or layer_names_to_plot is not None:
                import matplotlib.patches as patches

                for s_idx in target_steps:
                    for l_idx in target_layer_indices:
                        # Calculate position
                        x_pos = l_idx * W - 0.5
                        y_pos = s_idx * H - 0.5

                        # Add Red Border
                        rect = patches.Rectangle(
                            (x_pos, y_pos),
                            W,
                            H,
                            linewidth=1.5,
                            edgecolor="gold",
                            facecolor="none",
                            zorder=10,
                        )
                        ax.add_patch(rect)

            ax.set_ylabel("Computation Steps", fontsize=8, fontweight="bold")
            for idx, name in enumerate(all_layer_names):
                center_x = idx * W + (W / 2.0)
                lens_id = int(name[-1])

                ax.text(
                    center_x,
                    -H * 0.02,
                    f"L{lens_id + 1}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    transform=ax.transData,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Saved merged activations to {pdf_path}")

    def plot_layer_activations(
        self,
        traces: list[InferenceTrace],
        filename: str = "activations.pdf",
        step_id: int | None = None,
        layer_names_to_plot: list[str] | None = None,
    ):
        """Plot L2 energy over channels for multiple steps averaged over traces.

        If step_id and layer_names_to_plot provided, only those layers and that step are plotted.

        Args:
            traces: Inference traces
            filename: Pdf name. Defaults to "activations.pdf".
            step_id: Which step should be highlighted. Defaults to None.
            layer_names_to_plot: Which layers to plot. Defaults to None.
        """
        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        avg_energy_data, all_layer_names, n_steps = self._compute_averaged_energy_maps(traces)
        if not avg_energy_data:
            return

        first_trace = traces[0]

        target_steps = [step_id] if step_id is not None else range(n_steps)

        if layer_names_to_plot is not None:
            final_layers = [name for name in all_layer_names if name in layer_names_to_plot]
            if not final_layers:
                print(f"Warning: None of the requested layers {layer_names_to_plot} found in trace.")
                return
        else:
            final_layers = all_layer_names

        n_layers_plot = len(final_layers)

        with PdfPages(pdf_path) as pdf:
            for step_idx in target_steps:
                if step_idx >= n_steps:
                    continue

                step_maps = avg_energy_data[step_idx]
                current_step = first_trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step)

                fig, axes = plt.subplots(1, n_layers_plot, figsize=(3 * n_layers_plot, 1.5))
                if n_layers_plot == 1:
                    axes = [axes]

                for i, name in enumerate(final_layers):
                    ax = axes[i]
                    avg_map = step_maps[name]

                    vmin, vmax = 0.0, float(avg_map.max().item() if avg_map.numel() > 0 else 0.0)

                    ax.imshow(avg_map.numpy(), cmap=COLOR, vmin=vmin, vmax=vmax)
                    overlay_grid_text(ax, avg_map, grid_str, vmin, vmax)

                    layer_id = int(name[-1]) + 1
                    ax.set_title(f"L{layer_id}", fontsize=15)
                    ax.axis("off")

                    # --- Add Step ID Label to the left of the first plot ---
                    if i == 0:
                        ax.text(
                            -0.02,
                            0.5,  # x, y coordinates (negative x moves left)
                            f"Step {step_idx}",  # Text
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="right",
                            fontsize=15,
                            fontweight="bold",
                        )

                plt.tight_layout()
                pdf.savefig(fig)

                # Save frame for video if plotting full sequence
                if step_id is None:
                    frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                    fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                    frame_paths.append(frame_path)

                plt.close(fig)

        print(f"Saved activations to {pdf_path}")

        if frame_paths and step_id is None:
            try:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "5",
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(Path(temp_dir) / "frame_*.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    str(video_path),
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Saved activations video to {video_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            finally:
                shutil.rmtree(temp_dir)
        elif step_id is not None:
            shutil.rmtree(temp_dir)

    def plot_mask(self, trace: InferenceTrace, filename: str = "mask.pdf"):
        """
        Plots the attention mask from LensTransformer showing which positions are selected.
        Also saves as an MP4 video at 5fps.

        Args:
            trace: Inference trace with extra outputs containing mask.
            filename: Filename for the output PDF. Defaults to "mask.pdf".
        """
        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        h, w = self.dataset.h, self.dataset.w

        # Check if we have mask data
        if not trace.completion.steps:
            print("No steps found in trace")
            return

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(trace.n_steps - 1):
                current_step = trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step)

                # Get mask from the forward pass
                # We need to run a forward pass to get the mask
                with torch.no_grad():
                    x_in = trace.completion.steps[step_idx]
                    src_emb, tgt_emb = self.trainer.model.prepare_tokens(x_in, x_in)
                    _, _, _, _, _, extra = self.trainer.model(src_emb, tgt_emb)
                    mask = extra["mask"]  # (b, h*w, 1)

                # Reshape mask to grid
                mask_grid = rearrange(mask[0, :, 0], "(h w) -> h w", h=h, w=w).cpu()

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                # Plot mask heatmap
                im = ax.imshow(mask_grid, cmap=COLOR, vmin=0, vmax=1)

                # Overlay grid text
                overlay_grid_text(ax, mask_grid, grid_str, 0, 1)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Mask Value (0=masked, 1=selected)", rotation=270, labelpad=15)

                ax.set_title(f"Step {step_idx}: Attention Mask", fontsize=14, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])

                plt.tight_layout()

                # Save to PDF
                pdf.savefig(fig)

                # Save frame for video
                frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                frame_paths.append(frame_path)

                plt.close(fig)

        print(f"Saved mask visualization to {pdf_path}")

        # Create video using ffmpeg
        if frame_paths:
            try:
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    "5",
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(Path(temp_dir) / "frame_*.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    str(video_path),
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Saved mask video to {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to create video: {e.stderr.decode()}")
            except FileNotFoundError:
                print("Warning: ffmpeg not found. Install ffmpeg to generate video.")
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)

    def plot_attention_sensitivity(
        self,
        traces: list[InferenceTrace],
        filename: str = "attention_sensitivity.pdf",
        targets: list[tuple[int, int, int]] | None = None,
    ):
        """
        Plot sensitivity of attention weights w.r.t. input embeddings.
        Averages over multiple traces.

        Args:
            traces: List of InferenceTrace objects.
            filename: Output PDF filename.
            targets: List of (step_idx, layer_idx, head_idx) tuples to plot.
                     If None, plots ALL heads for ALL steps (generates video).
                     If provided, plots only the selected configurations side-by-side (no video).
        """
        if not traces:
            print("No traces provided.")
            return

        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        # Structural info from first trace
        first_trace = traces[0]
        n_steps = len(first_trace.completion.steps)
        h, w = self.dataset.h, self.dataset.w
        n_registers = self.trainer.head.n_registers

        # Determine mode: Specific Targets vs Full Video
        if targets is not None:
            # "Select specific heads" mode
            plot_steps = sorted(list(set(t[0] for t in targets)))
            mode = "targets"
        else:
            # "Plot everything" mode
            plot_steps = range(n_steps - 1)
            mode = "full"

        is_split = hasattr(self.trainer.head, "n_action_tokens")

        with torch.no_grad():
            x_dummy = first_trace.completion.steps[0]
            s_emb, t_emb = self.trainer.model.prepare_tokens(x_dummy, x_dummy)
            _, _, enc_attns_test, _, _, _ = self.trainer.model(s_emb, t_emb)
            n_layers_model = enc_attns_test.shape[0]
            n_heads_model = enc_attns_test.shape[2]

        self.trainer.model.eval()
        self.trainer.head.eval()

        # Storage for averaged sensitivities: { (step, layer, head): Tensor(h, w) }
        averaged_data = {}

        # We iterate by step to minimize forward passes
        for step_idx in plot_steps:
            if step_idx >= n_steps - 1:
                continue

            # Determine which (layer, head) pairs we need for this step
            if mode == "targets":
                relevant_pairs = [(t[1], t[2]) for t in targets if t[0] == step_idx]
            else:
                relevant_pairs = [(l, hd) for l in range(n_layers_model) for hd in range(n_heads_model)]

            if not relevant_pairs:
                continue

            # Accumulator for this step
            step_accum = {pair: [] for pair in relevant_pairs}

            for trace in traces:
                x_in = trace.completion.steps[step_idx]  # (1, h, w)

                # We need the (i, j) for this specific trace to know which query token to inspect
                current_ij_idx = trace.completion.indices[step_idx]  # tuple (i, j)
                flat_q_idx = n_registers + current_ij_idx[0] * w + current_ij_idx[1]

                with torch.set_grad_enabled(True):
                    src_emb, tgt_emb = self.trainer.model.prepare_tokens(x_in, x_in)
                    src_emb.retain_grad()

                    # Forward
                    _, _, enc_attns, _, _, _ = self.trainer.model(src_emb, tgt_emb)

                    if is_split:
                        n_action = self.trainer.head.n_action_tokens
                        enc_attns = enc_attns[..., n_action:, n_action:]

                    # Backward for each requested head
                    for lay, head in relevant_pairs:
                        attn_row = enc_attns[lay, 0, head, flat_q_idx, n_registers:]  # (h*w) keys

                        if src_emb.grad is not None:
                            src_emb.grad.zero_()

                        attn_row.backward(gradient=attn_row, retain_graph=True)

                        grad_norm = torch.norm(src_emb.grad[0], p=2, dim=-1).detach().cpu()
                        step_accum[(lay, head)].append(grad_norm)

            # Average for this step
            for pair, sens_list in step_accum.items():
                if sens_list:
                    stack = torch.stack(sens_list, dim=0)
                    avg = stack.mean(dim=0)
                    # Normalize
                    mx = avg.max()
                    if mx > 1e-8:
                        avg = avg / mx
                    averaged_data[(step_idx, pair[0], pair[1])] = avg

        with PdfPages(pdf_path) as pdf:

            if mode == "targets":
                # SINGLE PAGE, side-by-side
                n_plots = len(targets)
                fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 3.5, 1.8))
                if n_plots == 1:
                    axs = [axs]

                for idx, (s, l, h_idx) in enumerate(targets):
                    ax = axs[idx]
                    data = averaged_data.get((s, l, h_idx))

                    if data is None:
                        ax.text(0.5, 0.5, "No Data", ha="center")
                        ax.axis("off")
                        continue

                    # Get grid text from first trace
                    step_txt = first_trace.completion.steps[s][0]
                    grid_str = self.dataset.to_string(step_txt)

                    # Highlight current position
                    curr_ij = first_trace.completion.indices[s]

                    ax.imshow(data.numpy(), cmap=COLOR, vmin=0, vmax=1, aspect="equal")
                    overlay_grid_text(ax, data, grid_str, 0, 1, curr_ij)

                    ax.set_title(
                        f"Step {s}:  Layer {l+1}, Head {h_idx+1}",
                        fontsize=15,
                        fontweight="bold",
                    )
                    ax.axis("off")

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                print(f"Saved specific attention targets to {pdf_path}")

            else:
                # FULL VIDEO MODE (Page per step)
                for step_idx in plot_steps:
                    fig, axs = plt.subplots(
                        n_layers_model,
                        n_heads_model,
                        figsize=(n_heads_model * 3, n_layers_model * 3),
                        squeeze=False,
                    )

                    step_txt = first_trace.completion.steps[step_idx][0]
                    grid_str = self.dataset.to_string(step_txt)
                    curr_ij = first_trace.completion.indices[step_idx]

                    for l in range(n_layers_model):
                        for hd in range(n_heads_model):
                            ax = axs[l, hd]
                            data = averaged_data.get((step_idx, l, hd))

                            if data is not None:
                                ax.imshow(
                                    data.numpy(),
                                    cmap=COLOR,
                                    vmin=0,
                                    vmax=1,
                                    aspect="equal",
                                )
                                overlay_grid_text(ax, data, grid_str, 0, 1, curr_ij)

                            ax.set_title(f"L{l+1} H{hd+1}", fontsize=12, fontweight="bold")
                            ax.axis("off")

                    plt.suptitle(
                        f"Attention Sensitivity Step {step_idx}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    pdf.savefig(fig)

                    # Save frame
                    frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                    fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                    frame_paths.append(frame_path)
                    plt.close(fig)

                print(f"Saved full attention sensitivity to {pdf_path}")

                # Create video
                if frame_paths:
                    try:
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-y",
                            "-framerate",
                            "5",
                            "-pattern_type",
                            "glob",
                            "-i",
                            str(Path(temp_dir) / "frame_*.png"),
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-vf",
                            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            str(video_path),
                        ]
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        print(f"Saved attention video to {video_path}")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass
                    finally:
                        shutil.rmtree(temp_dir)

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

    def plot_input_sensitivity(
        self,
        traces: list[InferenceTrace],
        filename: str = "input_sensitivity.pdf",
        step_id: int | None = None,
    ):
        """
        Plot sensitivity of the final output prediction with respect to each input position.
        Merges all steps into a single grid layout on one PDF page.
        Averages gradients over multiple traces to show robust influence.

        Args:
            traces: List of InferenceTrace objects (must have same steps/dimensions).
            filename: Filename for the output PDF.
            step_id: If set, only plots sensitivity for this specific step index.
        """
        if not traces:
            print("No traces provided for sensitivity plotting.")
            return

        pdf_path = self.plot_dir / filename

        # Use the first trace for structural info
        first_trace = traces[0]
        n_steps = len(first_trace.completion.y_preds)
        h, w = self.dataset.h, self.dataset.w
        vocab_size = self.dataset.vocab_size()

        # Determine which steps to process
        if step_id is not None:
            target_steps = [step_id]
        else:
            target_steps = range(n_steps)

        num_plots = len(target_steps)
        if num_plots == 0:
            return

        # Determine Grid Dimensions
        import math

        ncols = math.ceil(math.sqrt(num_plots))
        nrows = math.ceil(num_plots / ncols)

        # Scale figure size based on grid and content
        # Base size per subplot approx 3x3 inches, scaling with grid width
        subplot_w = 2.5
        subplot_h = 1.5
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * subplot_w, nrows * subplot_h), squeeze=False)

        # Flatten axs for easy iteration
        axs_flat = axs.flatten()

        # Ensure model is in eval mode
        self.trainer.model.eval()
        self.trainer.head.eval()

        # Loop through target steps
        for plot_idx, step_idx in enumerate(target_steps):
            ax = axs_flat[plot_idx]

            if step_idx >= n_steps:
                ax.axis("off")
                continue

            # Accumulate sensitivity maps (H, W) for this step across all traces
            step_sensitivities = []

            i, j, v = 0, 0, 0

            for trace in traces:
                x_in = trace.completion.steps[step_idx]

                with torch.set_grad_enabled(True):
                    src_emb, tgt_emb = self.trainer.model.prepare_tokens(x_in, x_in)
                    src_emb.retain_grad()

                    enc_out, dec_out, _, _, _, _ = self.trainer.model(src_emb, tgt_emb)
                    current_out = self.trainer.head(dec_out if dec_out is not None else enc_out)

                    # Identify Target Logit
                    batch_idx = 0
                    if isinstance(current_out, tuple):
                        v_logits, loc_logits = current_out

                        loc_probs = torch.softmax(loc_logits, dim=-1)
                        v_probs = torch.softmax(v_logits, dim=-1)

                        best_loc = torch.argmax(loc_probs, dim=-1).item()
                        best_v = torch.argmax(v_probs, dim=-1).item()

                        target_logit = loc_logits[batch_idx, best_loc] + v_logits[batch_idx, best_v]

                        i = best_loc // w
                        j = best_loc % w
                        v = best_v
                    else:
                        output_reshaped = rearrange(current_out, "b (h w) v -> b h w v", h=h, w=w)
                        flat_out = current_out.view(current_out.size(0), -1)
                        max_idx = flat_out.argmax(dim=1).item()

                        best_loc_flat = max_idx // vocab_size
                        best_v = max_idx % vocab_size
                        i = best_loc_flat // w
                        j = best_loc_flat % w
                        v = best_v

                        target_logit = output_reshaped[batch_idx, i, j, v]

                    if src_emb.grad is not None:
                        src_emb.grad.zero_()

                    target_logit.backward()

                    input_grad = src_emb.grad
                    trace_sensitivity = torch.norm(input_grad[batch_idx], p=2, dim=-1).detach().cpu()
                    step_sensitivities.append(trace_sensitivity)

            if step_sensitivities:
                stacked_sens = torch.stack(step_sensitivities, dim=0)
                avg_sensitivity = stacked_sens.mean(dim=0).numpy()

                # Normalize
                norm_sensitivity = (avg_sensitivity - avg_sensitivity.min()) / (
                    avg_sensitivity.max() - avg_sensitivity.min() + 1e-8
                )

                ax.imshow(norm_sensitivity, cmap=COLOR, vmin=0, vmax=1, aspect="equal")

                # Overlay Text
                current_step_text = first_trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step_text)
                overlay_grid_text(ax, norm_sensitivity, grid_str, 0, 1, (i, j))

                ax.set_title(
                    f"Step {step_idx}",
                    fontsize=15,
                    fontweight="bold",
                )
            else:
                ax.text(0.5, 0.5, "No Data", ha="center")

            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for k in range(num_plots, len(axs_flat)):
            axs_flat[k].axis("off")

        plt.tight_layout()

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig)

        plt.close(fig)
        print(f"Saved input sensitivity grid to {pdf_path}")


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

    cfg.dataset.max_digits = args.n_digits
    dataset = instantiate(cfg.dataset)
    plotter = Plotter(dataset, trainer, model_dir)

    samples = [trainer.fabric.to_device(dataset.get_example(i)) for i in range(10)]
    pretty_print_sample(dataset, samples[0])

    # If no activation layers are provided, we pass None/Empty to InferenceTrace
    target_layers = args.activation_layers if args.activation_layers else []

    with torch.no_grad():
        # Initialize trace with the layers we want to monitor
        traces = [InferenceTrace(trainer, sample, target_layers) for sample in samples]

    if args.plot_probabilities:
        print("Plotting Probabilities...")
        if hasattr(trainer.head, "n_action_tokens"):
            plotter.plot_probabilities_split(traces[0])
        else:
            plotter.plot_probabilities_global(traces[0])

    if args.plot_attention:
        print("Plotting Attention...")
        plotter.plot_attention_analysis(traces[0])
    if args.activation_layers:
        print(f"Plotting Activations for: {args.activation_layers}")
        plot_step = 13
        plot_layers = ["_forward_module.lens.2", "_forward_module.lens.4"]
        # plot_step = None
        # plot_layers = None
        plotter.plot_layer_activations_merged(traces, step_id=plot_step, layer_names_to_plot=plot_layers)
        plotter.plot_layer_activations(traces, step_id=plot_step, layer_names_to_plot=plot_layers)

    if args.plot_conv_kernels:
        print("Plotting Convolution Kernels...")
        plotter.plot_conv_kernels()

    if args.semantic_kernel_layer:
        print(f"Plotting Semantic Kernels for: {args.semantic_kernel_layer}")
        plotter.plot_semantic_kernels(args.semantic_kernel_layer, trainer.model.src_embed)

    if args.plot_sensitivity:
        print("Plotting Input Sensitivity...")
        plotter.plot_input_sensitivity(traces)

    if args.plot_mask:
        print("Plotting Attention Mask...")
        plotter.plot_mask(traces[0])

    if args.plot_attention_sensitivity:
        print("Plotting Attention Sensitivity...")
        # heads = [(12, 1, 1), (12, 2, 2)]
        heads = None
        plotter.plot_attention_sensitivity(traces, targets=heads)


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

    parser.add_argument(
        "--n_digits",
        type=int,
        default=3,
        help="Number of digits for the dataset sample (default: 3)",
    )

    parser.add_argument(
        "--sensitivity",
        action="store_true",
        dest="plot_sensitivity",
        help="Enable input sensitivity plotting (gradient-based attribution)",
    )

    parser.add_argument(
        "--mask",
        action="store_true",
        dest="plot_mask",
        help="Enable attention mask plotting from LensTransformer",
    )

    parser.add_argument(
        "--attention-sensitivity",
        action="store_true",
        dest="plot_attention_sensitivity",
        help="Enable attention sensitivity plotting (gradient-based attribution)",
    )

    args = parser.parse_args()

    main(args)
