import pandas as pd

from utils.constants import DATASETS
from utils.utils import get_number_success

df_result = pd.read_excel('consolidated_results.xlsx', index_col=None)

for i, step in [['smote', 3], ['ADASYN', 6], ['OVERSAMP', 9], ['UNDERSAMP', 12]]:
    success = get_number_success(df_result, index=3, step=step)
    print('Gamma VS {}. Success: {} out of {}'.format(i, success, len(DATASETS) * 3))
