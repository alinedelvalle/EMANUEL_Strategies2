from sklearn.base import BaseEstimator, TransformerMixin

from mlfs_pyit.mdmr import mdmr

from sklearn.preprocessing import KBinsDiscretizer

import pandas as pd


class PyIT_MDMR(BaseEstimator, TransformerMixin):

    def __init__(self, n_features):
        self.n_features = int(n_features)
        self.index_ranks = None
        
        
    def __str__(self):
        return 'PYIT_MDMR('+str(self.n_features)+')'
    
    
    def fit(self, dfX, dfy=None):  
        # discretizer
        kb_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        kb_discretizer.fit(dfX)
        # a saída é um dataframe por conta da configuração: set_config(transform_output="pandas") 
        new_dfX = kb_discretizer.transform(dfX).astype(int) 
        X = new_dfX.to_numpy() 

        new_dfy = dfy.astype(int) 
        y = new_dfy.to_numpy() 

        self.index_ranks = mdmr().rank(X, y)
        
        return self
    

    def transform(self, dfX, dfy = None):
        indexes = self.index_ranks[:self.n_features]
        return dfX.iloc[:, indexes]