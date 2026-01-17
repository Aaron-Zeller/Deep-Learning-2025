#!/bin/bash

# Different num_locations

python3 train.py -cn lens_transformer_3_3_3_3_1_softmask_3_4_dig_splitaction_sepcarry_5loc ++n_epochs=200 &
python3 train.py -cn lens_transformer_3_3_3_3_1_softmask_3_4_dig_splitaction_sepcarry_8loc ++n_epochs=200 &

wait

