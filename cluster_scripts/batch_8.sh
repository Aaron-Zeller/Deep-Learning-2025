#!/bin/bash

# Separate Carry

python3 train.py -cn lens_transformer_3_3_3_3_1_softmask_1_2_dig_splitaction_sepcarry ++n_epochs=200 &
python3 train.py -cn lens_transformer_3_3_3_3_1_softmask_2_3_dig_splitaction_sepcarry ++n_epochs=200 &

wait

