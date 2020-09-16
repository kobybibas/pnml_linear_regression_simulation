# A New Look at an Old Problem: A Universal Learning Approach to Linear Regression
Koby Bibas, Yaniv Fogel and Meir Feder

https://arxiv.org/abs/1905.04708

### Abstract
Linear regression is a classical paradigm in statistics. A new look at it is provided via the lens of universal learning. In applying universal learning to linear regression the hypotheses class represents the label y as a linear combination of the feature vector x^Tθ, within a Gaussian error. The Predictive Normalized Maximum Likelihood (pNML) solution for universal learning of individual data can be expressed analytically in this case, as well as its associated learnability measure. Interestingly, the situation where the number of parameters M may even be larger than the number of training samples N can be examined. As expected, in this case learnability cannot be attained in every situation; nevertheless, if the test vector resides mostly in a subspace spanned by the eigenvectors associated with the large eigenvalues of the empirical correlation matrix of the training data, linear regression can generalize despite the fact that it uses an over-parametrized model. We demonstrate the results with a simulation of fitting a polynomial to data with a possibly large polynomial degree. 

<img src="https://raw.githubusercontent.com/kobybibas/pnml_linear_regression_isit_2019_latex/master/least_squares_with_poly_degree.jpg" width="750">

### Run the code

Install requiremnet

```bash
 pip install -r requirements.txt
```

execute simulation

```bash
 cd src
 python main.py --config_path ../configs/pnml_polynomial.yaml model_degree=3,6,9 -m
 # For minimum norm solution
 python main.py --config_path ../configs/pnml_min_norm_fourier model_degree=2,6,10 -m
```

Produce figures

```bash
cd notebooks
jupyter-notebook pnml_visualization.ipynb
```

The parameters for the code are located in configs directory

```bash
.
├── README.md
├── bash_scripts
├── configs
│   ├── pnml_fourier.yaml
│   ├── pnml_min_norm_fourier.yaml
│   └── pnml_polynomial.yaml
├── notebooks
│   ├── logistic_regression.ipynb
│   ├── min_norm_solution.ipynb
│   ├── notebook_utils.py
│   └── pnml_visualization.ipynb
├── output
├── requirements.txt
└── src
```


```bash
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=bostonHousing ; \
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=concrete; \
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=energy; \
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=kin8nm; \
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=naval-propulsion-plant; \
python main_real_data.py --config-name ../configs/uci_experiment.yaml  dataset_name=wine-quality-red;

```


### Citing
```
@inproceedings{bibas2019new,
  title={A new look at an old problem: A universal learning approach to linear regression},
  author={Bibas, Koby and Fogel, Yaniv and Feder, Meir},
  booktitle={2019 IEEE International Symposium on Information Theory (ISIT)},
  pages={2304--2308},
  year={2019},
  organization={IEEE}
}
```

