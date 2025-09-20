import numpy as np

import  joblib

from pymoo.core.problem import Problem

from multiobjective.IndividualUtils import IndividualUtils

from utils.HistoryRun import HistoryRun

from multiobjective.CodifyIndividualSurrogate import *

from multiprocessing.pool import ThreadPool 


class MLProblem(Problem):
    
    def __init__(self, n_gene, name_dataset, config, n_threads, path_model_f1, path_model_size, log_file, file_metrics):
        super().__init__(n_var=n_gene, n_obj=2) 
        
        self.name_dataset = name_dataset
        self.config = config
        self.n_threads = n_threads
        self.path_model1 = path_model_f1
        self.path_model2 = path_model_size
        self.log_file = log_file
        self.file_metrics = file_metrics
        
        self.n_ger = 0
        self.list_indexes_cols, self.dict_indexes = get_all_indexes(self.config)
        
    
    def to_file_ger(self):
        file = open(self.log_file, 'a')
        file.write(f'Geração: {str(self.n_ger)}\n')
        file.close()
    
    
    def my_eval(self, param):
        cmd, model1, model2 = param
        is_normalize, fp_command, meka_command, weka_command = cmd
        #print(is_normalize, fp_command, meka_command, weka_command)
        
        # obtém o algoritmo codificado e um dataframe de teste    
        alg_cod = codify(is_normalize, fp_command, meka_command, weka_command, self.dict_indexes, self.config) 
        x_test = pd.DataFrame([alg_cod], columns=self.list_indexes_cols)
        x_test = x_test + 1
        x_test.fillna(0, inplace=True)
        
        # F1
        f1 =  model1.predict(x_test)[0]
        
        # Model size
        log_model_size = model2.predict(x_test)
        model_size = np.exp(log_model_size)[0]
        
        #print(f1, model_size)
        HistoryRun.add_metrics(is_normalize, fp_command, meka_command, weka_command, f1, model_size)
        
        return [-f1, model_size]
    
        
    def _evaluate(self, X, out, *args, **kwargs):   
        models_f1 = joblib.load(self.path_model1)
        models_f2 = joblib.load(self.path_model2)

        F = [] 
        for k in range(len(X)):
            is_normalize, fp_command, meka_command, weka_command = IndividualUtils.get_commands(self.config, X[k])

            # obtém o algoritmo codificado e um dataframe de teste    
            alg_cod = codify(is_normalize, fp_command, meka_command, weka_command, self.dict_indexes, self.config) 
            x_test = pd.DataFrame([alg_cod], columns=self.list_indexes_cols)
            x_test = x_test + 1
            x_test.fillna(0, inplace=True)
            
            # F1
            f1 =  models_f1.predict(x_test)[0]
            
            # Model size
            log_model_size = models_f2.predict(x_test)
            model_size = np.exp(log_model_size)[0]
            
            HistoryRun.add_metrics(is_normalize, fp_command, meka_command, weka_command, f1, model_size)
            
            if len(F) == 0:
                F = [[-f1, model_size]]
            else:
                F.append([-f1, model_size]) 

        #print(F)

        # store the function values and return them.
        out["F"] = np.array(F, dtype=object)
        
        # log
        self.n_ger += 1 # geração
        self.to_file_ger() # armazena a geração corrente
        
        HistoryRun.to_file(self.name_dataset, self.file_metrics)