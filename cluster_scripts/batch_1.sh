#!/bin/bash

python3 train.py -cn lens_transformer_7_5_3_1_hardmask_proj_grad ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_hardmask_proj_grad_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_nomask_3_4_dig ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_nomask_3_4_dig_splitaction ++n_epochs=200 &

wait

python3 train.py -cn lens_transformer_7_5_3_1_nomask_proj_grad ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_nomask_proj_grad_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_7_5_3_1_softmask_3_4_dig ++n_epochs=200 &

wait