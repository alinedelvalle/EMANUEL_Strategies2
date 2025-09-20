import pandas as pd
import numpy as np

class HistoryRun :
    
    list_metrics = []
    id_general = 1
    flag = True
    
    title = ['id', 'normalize', 'feature_selection', 'meka', 'weka', 'F1 (macro averaged by label)', 'Model Size']
    
    # statistics is a dict
    def add_metrics(is_normalize, fp_command, meka_command, weka_command, f1, model_size):   
        metrics = []
        
        metrics.append(HistoryRun.id_general)
        metrics.append(is_normalize)
        #metrics.append('"'+fp_command+'"')
        metrics.append('"'+fp_command+'"')
        metrics.append(meka_command)
        metrics.append(weka_command)
        metrics.append(f1)
        metrics.append(model_size)
            
        HistoryRun.id_general = HistoryRun.id_general + 1
        HistoryRun.list_metrics.append(metrics)
        
        
    def to_file(dataset_name, file_name):        
        df = pd.DataFrame(data = HistoryRun.list_metrics, columns=HistoryRun.title)
            
        list_dataset = np.full((len(HistoryRun.list_metrics)), dataset_name)
        
        df.insert(1, 'Dataset', list_dataset)
        
        if HistoryRun.flag == True:
            df.to_csv(file_name, sep=';', index=False, mode='a')
        else:
            df.to_csv(file_name, sep=';', index=False, header=False, mode='a')
        
        HistoryRun.list_metrics = []
        HistoryRun.flag = False