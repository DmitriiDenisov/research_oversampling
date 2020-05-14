import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import sys
import pandas as pd
import numpy as np
import os

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.remove(os.path.join(project_path, 'utils'))
# sys.path.remove(os.path.join(project_path, 'utils'))
sys.path.append('../')

from utils.constants import DATASETS, MODES
from utils.utils import get_number_success

df_result = pd.read_excel('3.xlsx', index_col=None)
df_result = df_result.dropna(how='all').reset_index(drop=True)

for mode, step in [['smote', 4], ['ADASYN', 8], ['OVERSAMP', 12], ['UNDERSAMP', 16], ['smote+normal', 20]]:
    success = get_number_success(df_result, index=4, step=step, num_modes=7)
    print(f'Gamma VS {mode}. Success: {success} out of {len(DATASETS) * 4}')
