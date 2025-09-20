import pandas as pd
import numpy as np

class HistoryRun :
    
    list_metrics = []
    id_general = 1
    flag = True
    
    title = ['Id', 'K', 'Normalize', 'Feature Preprocessing', 'Meka', 'Weka', 'Accuracy', 'Jaccard index', 
             'Hamming score', 'Exact match', 'Jaccard distance', 'Hamming loss', 'ZeroOne loss', 'Harmonic score', 
             'One error', 'Rank loss', 'Avg precision', 'Log Loss (lim. L)', 'Log Loss (lim. D)', 'Micro Precision', 
             'Micro Recall', 'Macro Precision', 'Macro Recall', 'F1 (micro averaged)', 'F1 (macro averaged by label)', 
             'AUPRC (macro averaged)', 'AUROC (macro averaged)', 'Build Time', 'Test Time', 'Total Time', 'Accuracy (per label)', 
             'Harmonic (per label)', 'Precision (per label)', 'Recall (per label)', 'avg. relevance (test set)',
             'avg. relevance (predicted)', 'avg. relevance (difference)', 'Model Size', 'Features']
    
    # statistics is a dict
    def add_metrics(k, is_normalize, fp_command, meka_command, weka_command, statistics, model_size, features):   
        metrics = []
        
        metrics.append(HistoryRun.id_general)
        metrics.append(k)
        metrics.append(is_normalize)
        metrics.append('"'+fp_command+'"')
        metrics.append(meka_command)
        metrics.append(weka_command)
        
        if statistics != {}:
            metrics.append(float(statistics.get('Accuracy')))
            metrics.append(float(statistics.get('Jaccard index')))
            metrics.append(float(statistics.get('Hamming score')))
            metrics.append(float(statistics.get('Exact match')))
            metrics.append(float(statistics.get('Jaccard distance')))
            metrics.append(float(statistics.get('Hamming loss')))
            metrics.append(float(statistics.get('ZeroOne loss')))
            metrics.append(float(statistics.get('Harmonic score')))
            metrics.append(float(statistics.get('One error')))
            metrics.append(float(statistics.get('Rank loss')))
            metrics.append(float(statistics.get('Avg precision')))
            metrics.append(float(statistics.get('Log Loss (lim. L)')))
            metrics.append(float(statistics.get('Log Loss (lim. D)')))
            metrics.append(float(statistics.get('Micro Precision')))
            metrics.append(float(statistics.get('Micro Recall')))
            metrics.append(float(statistics.get('Macro Precision')))
            metrics.append(float(statistics.get('Macro Recall')))
            metrics.append(float(statistics.get('F1 (micro averaged)')))
            metrics.append(float(statistics.get('F1 (macro averaged by label)')))
            metrics.append(float(statistics.get('AUPRC (macro averaged)')))
            metrics.append(float(statistics.get('AUROC (macro averaged)')))
            metrics.append(float(statistics.get('Build Time')))
            metrics.append(float(statistics.get('Test Time')))
            metrics.append(float(statistics.get('Total Time')))
            # array
            metrics.append(statistics.get('Accuracy (per label)'))
            metrics.append(statistics.get('Harmonic (per label)'))
            metrics.append(statistics.get('Precision (per label)'))
            metrics.append(statistics.get('Recall (per label)'))
            metrics.append(statistics.get('avg. relevance (test set)'))
            metrics.append(statistics.get('avg. relevance (predicted)'))
            metrics.append(statistics.get('avg. relevance (difference)'))
            #
            metrics.append(model_size)
            #
            metrics.append(features)
        else:
            n = len(HistoryRun.title) - len(metrics)
            metrics = np.concatenate((metrics, np.empty(n)*pd.NA))
            
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