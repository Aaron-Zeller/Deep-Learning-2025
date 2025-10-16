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
```

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