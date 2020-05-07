#!/usr/bin/env bash
cd ../src

python main_min_norm.py constrain_factor=1.0 model_degree=2
python main_min_norm.py constrain_factor=1.0 model_degree=6
python main_min_norm.py constrain_factor=1.0 model_degree=10
