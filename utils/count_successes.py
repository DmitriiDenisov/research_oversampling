import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import pandas as pd
from utils.constants import DATASETS, MODES
from utils.utils_py import get_number_success

df_result = pd.read_excel('3.xlsx', index_col=None)
df_result = df_result.dropna(how='all').reset_index(drop=True)

for mode, step in [['smote', 4], ['ADASYN', 8], ['OVERSAMP', 12], ['UNDERSAMP', 16], ['smote+normal', 20]]:
    success = get_number_success(df_result, index=4, step=step, num_modes=7)
    print(f'Gamma VS {mode}. Success: {success} out of {len(DATASETS) * 4}')
