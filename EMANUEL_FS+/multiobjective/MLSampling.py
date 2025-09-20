import numpy as np

from pymoo.core.sampling import Sampling

class MLSampling(Sampling):
    
    
    def __init__(self, config, n_gene, is_dataset_sparse=False):
        super().__init__()
        self.config = config # configuração de hiperarâmetros
        self.n_gene = n_gene # número de genes
        self.is_dataset_sparse = is_dataset_sparse # se dataset é esparso
        
     
    # retorna a população com n_samples indivíduos    
    def _do(self, problem, n_samples, **kwargs):
        # seleciona normalização
        if self.is_dataset_sparse == False:
            norm = np.random.randint(2, size=n_samples)
        else:
            # dataset esparso não normaliza
            norm = np.random.randint(1, size=n_samples) # np.zeros(n_samples)
        
        # seleciona valores dos demais genes
        algs_hips = np.random.randint(self.config.get_seed(), size=(n_samples, self.n_gene - 1))
        
        # une os algoritmos e os hiperparâmetros selecionados
        X = np.column_stack([norm, algs_hips])
        #print(X)
        
        return X
