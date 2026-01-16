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
        Also saves as an MP4 video at 5fps.

        Args:
            trace: Inference trace with activations.
            filename: Filename for the output PDF. Defaults to "activations.pdf".
        """
        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        layer_names = list(trace.completion.activations[0].keys())  # assume same
        n_layers = len(layer_names)

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(trace.n_steps - 1):
                activations = trace.completion.activations[step_idx]
                current_step = trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step)

                fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 3))
                if n_layers == 0:
                    axes = [axes]

                for i, name in enumerate(layer_names):
                    ax = axes[i]
                    act = activations[name]  # (1, C, H, W)

                    energy_map = torch.norm(act, p=2, dim=1).squeeze()  # (H, W)

                    # Normalize energy map for visualization
                    vmin, vmax = 0, energy_map.max().item()

                    im = ax.imshow(energy_map, cmap=COLOR, vmin=vmin, vmax=vmax)

                    # Overlay grid text on activation map
                    overlay_grid_text(ax, energy_map.cpu(), grid_str, vmin, vmax)

                    ax.set_title(f"{name}\n(L2 Energy)")
                    ax.axis("off")

                plt.suptitle(f"Layer Activations at Step {step_idx}", fontsize=16)
                plt.tight_layout()

                # Save to PDF
                pdf.savefig(fig)

                # Save frame for video
                frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                frame_paths.append(frame_path)

                plt.close(fig)

        print(f"Saved activations to {pdf_path}")

        # Create video using ffmpeg
        if frame_paths:
            try:
                # Use ffmpeg to create video at 5fps
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-framerate",
                    "5",  # Input framerate
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(Path(temp_dir) / "frame_*.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure dimensions are even
                    str(video_path),
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Saved activations video to {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to create video: {e.stderr.decode()}")
            except FileNotFoundError:
                print("Warning: ffmpeg not found. Install ffmpeg to generate video.")
            finally:
                # Clean up temporary files
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

    def plot_attention_sensitivity(self, trace: InferenceTrace, filename: str = "attention_sensitivity.pdf"):
        """Plot sensitivity of attention weights w.r.t. input embeddings.

        Computes gradients of attention patterns from the current prediction
        position with respect to input embeddings, scaled by attention values.
        This reveals what input regions influence the attention mechanism after
        the lens CNN transformation.

        Args:
            trace: Inference trace containing model predictions and steps.
            filename: Output PDF filename. Also creates corresponding .mp4 video.
        """
        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        n_registers = self.trainer.head.n_registers
        h, w = self.dataset.h, self.dataset.w
        n_steps = trace.n_steps

        # Get attention info from first step
        with torch.no_grad():
            x_in_test = trace.completion.steps[0]
            src_emb_test, tgt_emb_test = self.trainer.model.prepare_tokens(x_in_test, x_in_test)
            _, _, enc_attns_test, _, _, _ = self.trainer.model(src_emb_test, tgt_emb_test)
            n_layers = enc_attns_test.shape[0]
            n_heads = enc_attns_test.shape[2]

        # Check for split head architecture
        is_split = hasattr(self.trainer.head, "n_action_tokens")

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(n_steps - 1):  # last step doesn't have attention
                print(f"Processing attention sensitivity for step {step_idx + 1}/{n_steps - 1}...")

                current_step = trace.completion.steps[step_idx][0]
                grid_str = self.dataset.to_string(current_step)
                i, j = trace.completion.indices[step_idx]

                # Forward pass with gradients enabled
                with torch.set_grad_enabled(True):
                    self.trainer.model.eval()
                    self.trainer.head.eval()

                    x_in = trace.completion.steps[step_idx]
                    src_emb, tgt_emb = self.trainer.model.prepare_tokens(x_in, x_in)
                    src_emb.retain_grad()

                    enc_out, _, enc_attns, _, _, extra = self.trainer.model(src_emb, tgt_emb)

                    # Handle split head architecture
                    if is_split:
                        n_action_tokens = self.trainer.head.n_action_tokens
                        enc_attns = enc_attns[..., n_action_tokens:, n_action_tokens:]

                    # Create figure with size adjusted for grid dimensions
                    # Scale subplot size based on grid width for better visibility
                    subplot_width = max(3, w * 0.5)  # At least 3 inches, scales with width
                    subplot_height = max(3, h * 0.5)  # At least 3 inches, scales with height
                    fig, axs = plt.subplots(
                        n_layers,
                        n_heads,
                        figsize=(n_heads * subplot_width, n_layers * subplot_height),
                        squeeze=False,
                    )

                    for lay in range(n_layers):
                        for head in range(n_heads):
                            # Get attention from position (i, j) to all other positions
                            attn_from_ij = enc_attns[lay, 0, head, n_registers + i * w + j, n_registers:]

                            # Compute sensitivity map
                            sensitivity_map = torch.zeros(h, w, device=src_emb.device)

                            for k in range(h * w):
                                attn_weight = attn_from_ij[k]

                                # Skip if attention weight is very small
                                if attn_weight.item() < 1e-6:
                                    continue

                                # Compute gradient
                                attn_weight.backward(retain_graph=True)

                                # Get gradient and compute L2 norm across embedding dimension
                                grad_norm = torch.norm(src_emb.grad[0], p=2, dim=-1)  # (h, w)

                                # Scale by attention value and accumulate
                                sensitivity_map += attn_weight.item() * grad_norm

                                # Clear gradients for next iteration
                                src_emb.grad.zero_()

                            # Normalize for visualization
                            max_val = sensitivity_map.max()
                            if max_val > 1e-8:
                                sensitivity_map = sensitivity_map / max_val
                            else:
                                sensitivity_map = torch.zeros_like(sensitivity_map)

                            # Plot
                            ax = axs[lay, head]
                            im = ax.imshow(sensitivity_map.cpu(), cmap=COLOR, vmin=0, vmax=1, aspect='equal')

                            # Overlay grid text
                            overlay_grid_text(ax, sensitivity_map.cpu(), grid_str, 0, 1, (i, j))

                            # Formatting
                            title_fontsize = max(10, min(14, 12 - w // 15))  # Adjust font size for wider grids
                            ax.set_title(f"Layer {lay+1} Head {head+1}", fontsize=title_fontsize, fontweight='bold')
                            ax.set_xticks([])
                            ax.set_yticks([])

                            # Add colorbar with adjusted size
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cbar = fig.colorbar(im, cax=cax)
                            cbar.ax.tick_params(labelsize=9)

                # Adjust title font size for wider grids
                title_fontsize = max(14, min(18, 18 - w // 15))
                plt.suptitle(
                    f"Attention Sensitivity at Step {step_idx}, Position ({i},{j})",
                    fontsize=title_fontsize,
                    fontweight='bold',
                )
                plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle

                # Save to PDF
                pdf.savefig(fig)

                # Save frame for video
                frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                frame_paths.append(frame_path)

                plt.close(fig)

        print(f"Saved attention sensitivity to {pdf_path}")

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
                print(f"Saved attention sensitivity video to {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to create video: {e.stderr.decode()}")
            except FileNotFoundError:
                print("Warning: ffmpeg not found. Install ffmpeg to generate video.")
            finally:
                # Clean up temporary files
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

    def plot_input_sensitivity(self, trace: InferenceTrace, filename: str = "input_sensitivity.pdf"):
        """Plot sensitivity of the final output prediction with respect to each input position.
        Uses gradient-based attribution to show which input positions most influence the prediction.
        Also saves as an MP4 video at 5fps.

        Args:
            trace: Inference trace containing model predictions.
            filename: Filename for the output PDF. Defaults to "input_sensitivity.pdf".
        """
        pdf_path = self.plot_dir / filename
        video_path = self.plot_dir / filename.replace(".pdf", ".mp4")

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        steps = trace.completion.steps
        y_preds = trace.completion.y_preds
        n_steps = len(y_preds)
        h, w = self.dataset.h, self.dataset.w
        vocab_size = self.dataset.vocab_size()

        with PdfPages(pdf_path) as pdf:
            for step_idx in range(n_steps):
                current_step = steps[step_idx][0]
                y_pred = y_preds[step_idx]

                # Get the predicted output cell
                if isinstance(y_pred, tuple):
                    # Split head case: (val_logits, pos_logits)
                    v_logits, loc_logits = y_pred
                    y_pred_flat = torch.zeros(v_logits.shape[0], h * w * vocab_size, device=v_logits.device)

                    # Compute outer product to get full distribution
                    loc_probs = torch.softmax(loc_logits, dim=-1)  # (b, h*w)
                    v_probs = torch.softmax(v_logits, dim=-1)  # (b, vocab)

                    for b in range(v_logits.shape[0]):
                        for pos_idx in range(h * w):
                            for v_idx in range(vocab_size):
                                flat_idx = pos_idx * vocab_size + v_idx
                                y_pred_flat[b, flat_idx] = loc_probs[b, pos_idx] * v_probs[b, v_idx]
                else:
                    # Global head case
                    y_pred_flat = rearrange(y_pred, "b (h w) v -> b (h w v)", h=h, w=w)

                max_indices = y_pred_flat.argmax(dim=1)
                batch_idx = 0
                i, j, v = torch.unravel_index(max_indices[batch_idx : batch_idx + 1], (h, w, vocab_size))
                i, j, v = i.item(), j.item(), v.item()

                # Run forward pass with gradients enabled
                x_in = steps[step_idx]

                # Enable gradient computation
                with torch.set_grad_enabled(True):
                    self.trainer.model.eval()  # Keep in eval to avoid dropout/etc
                    self.trainer.head.eval()

                    # Prepare embeddings
                    src_emb, tgt_emb = self.trainer.model.prepare_tokens(x_in, x_in)

                    # Retain gradients on source embeddings
                    src_emb.retain_grad()

                    # Forward pass through model
                    enc_out, dec_out, enc_attns, dec_self_attns, dec_cross_attns, extra = self.trainer.model(
                        src_emb, tgt_emb
                    )

                    # Head prediction
                    y_pred = self.trainer.head(dec_out if dec_out is not None else enc_out)

                    # Get the logit at the predicted position
                    if isinstance(y_pred, tuple):
                        v_logits, loc_logits = y_pred
                        target_logit = loc_logits[batch_idx, i * w + j] + v_logits[batch_idx, v]
                    else:
                        output_reshaped = rearrange(y_pred, "b (h w) v -> b h w v", h=h, w=w)
                        target_logit = output_reshaped[batch_idx, i, j, v]

                    # Compute gradients with respect to source embeddings
                    target_logit.backward()

                    # Get gradients with respect to input embeddings
                    input_grad = src_emb.grad

                # Aggregate gradients per position (L2 norm across embedding dimension)
                sensitivity = torch.norm(input_grad[batch_idx], p=2, dim=-1).cpu().numpy()

                # Normalize for visualization
                sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-8)

                # Create visualization with adaptive sizing
                fig_width = max(6, w * 0.8)  # Scale with grid width
                fig_height = max(6, h * 0.8)  # Scale with grid height
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

                # Plot sensitivity heatmap
                im = ax.imshow(sensitivity, cmap=COLOR, vmin=0, vmax=1, aspect='equal')

                # Overlay grid text
                grid_str = self.dataset.to_string(current_step)
                overlay_grid_text(ax, sensitivity, grid_str, 0, 1, (i, j))

                # Add colorbar with better layout
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label("Sensitivity (normalized)", rotation=270, labelpad=15, fontsize=11)
                cbar.ax.tick_params(labelsize=10)

                # Title with adaptive font size
                predicted_token = self.dataset.token_to_idx[v]
                title_fontsize = max(12, min(16, 16 - w // 15))
                ax.set_title(
                    f"Step {step_idx}: Input Sensitivity\nPredicted: '{predicted_token}' at ({i},{j})",
                    fontsize=title_fontsize,
                    fontweight="bold",
                )
                ax.set_xticks([])
                ax.set_yticks([])

                plt.tight_layout()
                pdf.savefig(fig)

                # Save frame for video
                frame_path = Path(temp_dir) / f"frame_{step_idx:04d}.png"
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                frame_paths.append(frame_path)

                plt.close(fig)

        print(f"Saved input sensitivity to {pdf_path}")

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
                print(f"Saved input sensitivity video to {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to create video: {e.stderr.decode()}")
            except FileNotFoundError:
                print("Warning: ffmpeg not found. Install ffmpeg to generate video.")
            finally:
                # Clean up temporary files
                shutil.rmtree(temp_dir)


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

    sample = trainer.fabric.to_device(dataset.get_example(0))
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

    if args.plot_sensitivity:
        print("Plotting Input Sensitivity...")
        plotter.plot_input_sensitivity(trace)

    if args.plot_mask:
        print("Plotting Attention Mask...")
        plotter.plot_mask(trace)

    if args.plot_attention_sensitivity:
        print("Plotting Attention Sensitivity...")
        plotter.plot_attention_sensitivity(trace)


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
