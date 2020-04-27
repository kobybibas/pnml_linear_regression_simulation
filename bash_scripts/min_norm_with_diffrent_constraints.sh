#!/usr/bin/env bash
cd ../src

model_degree=12

python main_min_norm.py constrain_factor=0.25 model_degree=${model_degree}
python main_min_norm.py constrain_factor=0.5 model_degree=${model_degree}
python main_min_norm.py constrain_factor=0.75 model_degree=${model_degree}
python main_min_norm.py constrain_factor=1.0 model_degree=${model_degree}
python main_min_norm.py constrain_factor=1.25 model_degree=${model_degree}
python main_min_norm.py constrain_factor=1.5 model_degree=${model_degree}
python main_min_norm.py constrain_factor=1.75 model_degree=${model_degree}
python main_min_norm.py constrain_factor=2.0 model_degree=${model_degree}