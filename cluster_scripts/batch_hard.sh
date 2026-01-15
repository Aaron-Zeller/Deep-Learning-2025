#!/bin/bash

python3 train.py -cn 33331_hard_mix_split_sep ++n_epochs=200 &
python3 train.py -cn 33331_hard_nomix_split_sep ++n_epochs=200 &
python3 train.py -cn 33331_hard_mix_global_sep ++n_epochs=200 &
python3 train.py -cn 7531_hard_mix_split_sep ++n_epochs=200 &

wait