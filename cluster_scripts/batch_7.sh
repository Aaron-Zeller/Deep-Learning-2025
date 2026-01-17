#!/bin/bash

python3 train.py -cn lens_transformer_3_3_3_3_1_nomask_3_4_dig_splitaction_sepcarry ++n_epochs=200 &

wait
