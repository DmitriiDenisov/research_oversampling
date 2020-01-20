## Oversampling research

## TODO:
- [x] Draw graphs of Gamma Distribution in Desmos, examine parameters

https://www.desmos.com/calculator/gxk5zydswy

As a test values `k=1` and `thehta=3` will be taken 

- [x] How to make random variable with Gamma Distribution


Available in `Research Gamma Dist.ipynb`
However still question about section, because Gamma distribution generates random variable on [0;+oo] meanwhile we need only for [0;1]

- [x] How to generate random variable with Gamma Distribution on a section

- [x] Generate new point for two points on 2D

- [x] Generate multiple points for two points on 2D

- [x] Visualization for above on 2D

- [x] Write test function for testing whether three points are on one line or not

- [x] Generate new point for two points for n dimentional space

- [x] Generate multiple points for two points for n dimentional space

- [x] Generalize this approach for n minority points for n dimentional space

Currently: randomly choose any two minority points, generate new point on the line between than. Repeat this process n times. Selected pair of minority points can repeat

- [x] Implement another approach for Gamma distribution when there is probability to generate new point on the other side

- [x] Add cross-validation

- [x] Add loop over all datasets and creating resulting DataFrame

- [x] Understand how is implemeted SMOTE in `imblearn` library:
https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

- [x] Add implementation of k_neighbours parameter for Gamma dist

- [x] Add realisation of SMOTE in order to compare

- [x] Add to statistics dataframe number of elements and percantage of minority class

- [x] Debug code to check everything

- [ ] Find random state and parameters for achieving 33-34 number of successes 
seed=0, [[1, 2], [1, 2.5]], successes=[30, 28] 
seed=0, [[1, 2], [1, 2.5]], successes=[33, 24]

- [ ] Add random oversampling and random undesampling

- [ ] Add implemetation of ADASYN


## Materials on SMOTE

https://basegroup.ru/community/articles/imbalance-datasets

https://youtu.be/FheTDyCwRdE


## Possible issues:

1. 

Problem:
`from imblearn.datasets import fetch_datasets` does not work for me as inside it fails in importing `from sklearn.utils.fixes import makedirs`

Solution:
Forcing `pip install imbalanced-learn==0.5.0`. The problem was observed on 0.4 version

