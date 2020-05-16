import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import pandas as pd
from utils.constants import DATASETS, MODES
from utils.utils_py import get_number_success

df_result = pd.read_excel('5.xlsx', index_col=None)
df_result = df_result.dropna(how='all').reset_index(drop=True)

all_success = 0
list_comp = [['smote', 4], ['ADASYN', 8], ['OVERSAMP', 12], ['UNDERSAMP', 16], ['smote+normal', 20]]
for mode, step in list_comp:
    success = get_number_success(df_result, index=4, step=step, num_modes=7)
    all_success += success
    print(f'Gamma VS {mode}. Success: {success} out of {len(DATASETS) * 4}')
print(f'Total number of success:{all_success} out of {len(list_comp) * len(DATASETS) * 4}')
