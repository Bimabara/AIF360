import os

import pandas as pd
import numpy as np

from aif360.sklearn.datasets.utils import standardize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')

STUDENT_PERFORMANCE_MAT_URL = 'https://raw.githubusercontent.com/Bimabara/dataset/master/student-mat.csv'


def fetch_student_performance(*, data_home=None, cache=True,
                   usecols=['school', 'sex', 'age', 'address', 'famsize',
                            'Pstatus', 'Medu', 'Fedu',
                            'Mjob', 'Fjob',
                            'reason', 'guardian', 'traveltime',
                            'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                            'activities', 'nursery', 'higher', 'internet',
                            'romantic', 'famrel', 'freetime', 'goout',
                            'Dalc', 'Walc', 'health', 'absences',
                            'G1', 'G2', 'G3'],
                   dropcols=['weight', 'payer_code'], numeric_only=False, dropna=True):
    """Load the Student Performance dataset.
    Args:
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
	@@ -44,59 +41,34 @@ def fetch_diabetes(subset='all', *, data_home=None, cache=True, binary_race=Fals
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.
    """
    
    data_url = STUDENT_PERFORMANCE_MAT_URL
    

    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                                os.path.basename(data_url))
    if cache and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col='encounter_id')
    else:
        df = pd.read_csv(data_url, index_col='encounter_id')
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
        
        # remap null values to np.NaN instead of '?'
        df = df.replace({'?': np.NaN})
        
        for col in df.select_dtypes(include=object):
            df[col] = df[col].fillna(df[col].mode()[0])
        
        
        return standardize_dataset(df, prot_attr=['sex', 'age'],
                               target='student-achievement', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)
