#!/bin/bash

# 1D vs. 2D

python3 train.py -cn transformer_rope1d_12d ++n_epochs=200 &
python3 train.py -cn transformer_freq_2d ++n_epochs=200 &

wait

