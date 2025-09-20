# ABC - Abstratct Base Class
from abc import ABC, abstractmethod

class FSAlgorithms(ABC):

    def select(self, dfX, dfy, n_features):
        index_ranks = self.rank(dfX, dfy)

        selecteds = index_ranks[:n_features]
        
        return selecteds
    
    
    @abstractmethod
    def rank(self, dfX, dfy):
        pass