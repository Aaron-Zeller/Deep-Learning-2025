#!/bin/bash

python3 train.py -cn lens_transformer_3_3_1_softmask_3_4_dig_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_3_3_3_1_softmask_3_4_dig_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_5_3_1_softmask_3_4_dig_splitaction ++n_epochs=200 &
python3 train.py -cn lens_transformer_5_3_3_1_softmask_3_4_dig_splitaction ++n_epochs=200 &

wait
