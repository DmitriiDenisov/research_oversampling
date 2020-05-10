## Oversampling research

## How to run:



## Materials on SMOTE

https://basegroup.ru/community/articles/imbalance-datasets

https://youtu.be/FheTDyCwRdE

## Authors:
Firuz Kamalov
Dmitry Denisov

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
