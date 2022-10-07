import os

import pandas as pd

from aif360.datasets import StandardDataset


class StudentPerformanceData(StandardDataset):
    
    def __init__(self, label_name='student-achievement', 
                 favorable_classes=['G1','G2','G3'],
                 protected_attribute_names=['sex', 'age',],
                 privileged_classes=[],
                 instance_weights_name=None,
                 categorical_features=['address', 'famsize', 'Pstatus', 'Mjob',
                     'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup ',
                     'paid ', 'activities ','nursery ', 'higher ', 'internet ',
                     'romantic '],
                 features_to_keep=[], 
                 features_to_drop=[],
                 na_values=["unknown"], 
                 custom_preprocessing=None,
                 metadata=None):
        
        filepath = os.path.dirname(os.path.abspath(__file__)),
        
        try:
            df = pd.read_csv(filepath, sep=';', na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'bank'))))
            import sys
            sys.exit(1)
        
        super(StudentPerformanceData, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
    
    
