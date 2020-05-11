# Oversampling research

## How to run:

Run:
`python3 main.py`

This will produce a file `compare_temp/output_{k}_{theta}_success_{success}_seed_{seed_value}.xlsx`

Execution time on server ~ 1 hour

## Dependencies:

`pip install -r requirements.txt`

## Structure of Repo:

`main.py` - script which runs all experiments and produces results into excel file
 
`test.py` - script for tests implemented with `pytest` library.
 
 To run: `pytest tests.py -vv` 
 
 `utils` folder contains secondary functions which are used in `main.py`
 
 `tests` folder contains functions only for tests which are used in `test.py`
 
 `experiments` contains several jupyter notebooks only for visualization and experiential purposes

## Authors:

Firuz Kamalov firuz@cud.ac.ae

Dmitry Denisov dmitryhse@gmail.com

## Possible issues:

1. 

Problem:
`from imblearn.datasets import fetch_datasets` does not work for me as inside it fails in importing `from sklearn.utils.fixes import makedirs`

Solution:
Forcing `pip install imbalanced-learn==0.5.0`. The problem was observed on 0.4 version

2. 

Problem:
`urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed`

Solution:
https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org

## Additional on SMOTE and ADDASYN

https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py

https://youtu.be/FheTDyCwRdE

https://basegroup.ru/community/articles/imbalance-datasets
