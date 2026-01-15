#!/bin/bash

python3 train.py -cn lens_transformer_7_5_3_1_softmask_3_4_dig_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_softmask_proj_grad ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_softmask_proj_grad_splitaction ++n_epochs=200 &
python3 train.py -cn transformer_freq_1d ++n_epochs=200 &

wait

python3 train.py -cn transformer_freq_1d_splitaction ++n_epochs=200 &
python3 train.py -cn transformer_rope_12d ++n_epochs=200 &
python3 train.py -cn transformer_rope_12d_splitaction ++n_epochs=200 &

wait