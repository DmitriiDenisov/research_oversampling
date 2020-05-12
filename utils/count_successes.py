import pandas as pd

from utils.constants import DATASETS
from utils.utils import get_number_success

df_result = pd.read_excel('consolidated_results.xlsx', index_col=None)
df_result = df_result.dropna(how='all')

for mode, i in [['smote', 4], ['ADASYN', 8], ['OVERSAMP', 12], ['UNDERSAMP', 16]]:
    success = get_number_success(df_result, index=i, num_modes=7)
    print(f'Gamma VS {mode}. Success: {success} out of {len(DATASETS) * 4}')
