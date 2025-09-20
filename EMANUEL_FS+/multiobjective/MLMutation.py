import numpy as np

from pymoo.core.mutation import Mutation

from multiobjective.IndividualUtils import IndividualUtils


class MLMutation(Mutation):
    
    def __init__(self, prob, config):
        super().__init__()
        self.prob = prob
        self.config = config


    def _do(self, problem, X, **kwargs):
        
        for individual in X:
            
            n_rand = np.random.rand()  
            
            # Are there mutation?
            if (n_rand < self.prob):

                #print(n_rand)
                #print(f'{IndividualUtils.get_commands(self.config, individual)}\n')
                
                # index of mutation
                len_individual = IndividualUtils.get_lenght_individual(self.config, individual)
                index = np.random.randint(0, len_individual)
                
                # new vaue of gene
                value = np.random.randint(0, self.config.get_seed())
                individual[index] = value

                #print(f'{IndividualUtils.get_commands(self.config, individual)}\n---')
                
        return X