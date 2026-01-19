# Transformers as Algorithmic Reasoners: A 2D Blackboard Approach

<div align="center">
Cyril Moser
&nbsp;&nbsp;&nbsp;&nbsp;
Gent Serifi
&nbsp;&nbsp;&nbsp;&nbsp;
Nicola Studer
&nbsp;&nbsp;&nbsp;&nbsp;
Aaron Zeller
&nbsp;&nbsp;&nbsp;&nbsp;

ETH Zurich, Switzerland
</div>

## Get Started

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip # Good practice to update pip
```

### Install PyTorch and PyTorch Lightning

```bash
# Should work for Apple Silicon and CUDA 12.8
# For other versions see https://pytorch.org/get-started/locally/
pip install torch torchvision lightning
```

### Install Remaining Dependencies

```bash
pip install -e .
```

### Setup Pre-Commit hooks (Development only)

```bash
pre-commit install
```

This will run `black` code formatting to ensure consistency. Note that if the check fails, the commit will be rejected. You can also run `black .` prior to committing to bring the code into the right shape.

## Train

```bash
python3 train.py
# More examples:
python3 train.py name=myexperiment
python3 train.py model=encoder_only_small resume=true # finds latest checkpoint
python3 train.py model=encoder_only_small resume_ckpt=outputs/exp/2025-11-16/16-47-17/checkpoints/ckpt_15.pth
python3 train.py model=encoder_only_small dataset.n_samples=1000
```

## Tensorboard

```bash
tensorboard --logdir outputs/
```

## Plotting

```
python plotting/plotter.py \
--model_dir outputs/lens_transformer_3_3_3_3_1_nomask_proj_grad/2025-12-23/18-38-22 \
  --ckpt ckpt_0120.pth \
  --activation_layers \
      _forward_module.lens.0 \
      _forward_module.lens.3 \
      _forward_module.lens.5 \
      _forward_module.lens.7 \
      _forward_module.lens.8 \
  --semantic_kernel_layer _forward_module.lens.0
```

The full reference of arguments can be acquired via `python plotting/plotter.py --help`.

## References

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

```bibtex
@article{shazeer2020glu,
  title={Glu variants improve transformer},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:2002.05202},
  year={2020}
}
```

```bibtex
@article{su2024roformer,
  title={Roformer: Enhanced transformer with rotary position embedding},
  author={Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  journal={Neurocomputing},
  volume={568},
  pages={127063},
  year={2024},
  publisher={Elsevier}
}
```
