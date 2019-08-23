# A New Look at an Old Problem: A Universal Learning Approach to Linear Regression
Koby Bibas, Yaniv Fogel and Meir Feder

https://arxiv.org/abs/1905.04708

### Abstract
Linear regression is a classical paradigm in statistics. A new look at it is provided via the lens of universal learning. In applying universal learning to linear regression the hypotheses class represents the label y as a linear combination of the feature vector x^Tθ, within a Gaussian error. The Predictive Normalized Maximum Likelihood (pNML) solution for universal learning of individual data can be expressed analytically in this case, as well as its associated learnability measure. Interestingly, the situation where the number of parameters M may even be larger than the number of training samples N can be examined. As expected, in this case learnability cannot be attained in every situation; nevertheless, if the test vector resides mostly in a subspace spanned by the eigenvectors associated with the large eigenvalues of the empirical correlation matrix of the training data, linear regression can generalize despite the fact that it uses an over-parametrized model. We demonstrate the results with a simulation of fitting a polynomial to data with a possibly large polynomial degree. 


<img src="https://raw.githubusercontent.com/kobybibas/pnml_linear_regression_isit_2019_latex/master/least_squares_with_poly_degree.jpg" width="750">

### Run the code

The parameters for the code are loacted in src/params.json
To run the code

1. clone the repository
2. pip install -r requirements.txt
3. python src/main.py
4. jupyter-notebook notebooks/analyze_results.ipynb


In order to run the ISIT 2019 figures:
1. git checkout isit
2. python src/main_isit.py

### Citing
```
@article{bibas2019new,
  title={A New Look at an Old Problem: A Universal Learning Approach to Linear Regression},
  author={Bibas, Koby and Fogel, Yaniv and Feder, Meir},
  journal={arXiv preprint arXiv:1905.04708},
  year={2019}
}
```




