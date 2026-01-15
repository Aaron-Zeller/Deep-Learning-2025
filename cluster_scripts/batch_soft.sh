#!/bin/bash

python3 train.py -cn 33331_soft_mix_split_sep ++n_epochs=200 &
python3 train.py -cn 33331_soft_nomix_split_sep ++n_epochs=200 &
python3 train.py -cn 33331_soft_mix_global_sep ++n_epochs=200 &
python3 train.py -cn 7531_soft_mix_split_sep ++n_epochs=200 &

wait