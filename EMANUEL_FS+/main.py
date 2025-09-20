import sys

import os
import pathlib

import numpy as np

import argparse

from configuration.Configuration import Configuration
from multiobjective.MLProblem import MLProblem
#from multiobjective.MLProblem2 import MLProblem
from multiobjective.MLSampling import MLSampling
from multiobjective.MLMutation import MLMutation
from multiobjective.IndividualUtils import IndividualUtils

from utils.Graphic import Graphic
from utils.ManipulateHistory import ManipulateHistory
from utils.HistoryRun import HistoryRun

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover

from pymoo.core.termination import TerminateIfAny
from pymoo.core.duplicate import NoDuplicateElimination

from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination


def create_dir(destination, name_dir):    
    cmd = 'if [ ! -d '+destination+'/'+name_dir+' ]\nthen\n\tmkdir '+destination+'/'+name_dir+'\nfi'
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_labels', type=int, required=False, default=15)
    parser.add_argument('--n_features', type=int, required=False, default=260)
    parser.add_argument('--name_dataset', type=str, required=False, default='birds')
    parser.add_argument('--is_dataset_sparse', action='store_true', required=False, default=False) # para true utiliza --is_dataset_sparse
    parser.add_argument('--len_population', type=int, required=False, default=100)
    parser.add_argument('--number_generation', type=int, required=False, default=100)
    parser.add_argument('--project', type=str, required=False, default= '/media/dellvale/Dados/Doutorado/Experimentos/Experimento5/EmanuelPlus')
    parser.add_argument('--name_dir_res', type=str, required=False, default='exe5')

    args = parser.parse_args() 

    print(args.name_dataset, args.is_dataset_sparse)
    
    n_threads = 1 
    n_gene = 40 # do indivíduo
    termination_period = 10
    termination_tol = 0.001 # de melhora
    
    # log
    log_file = args.project+'/log/'+'log_'+args.name_dataset+'_'+args.name_dir_res+'.txt'
    
    # pasta de resultados
    create_dir(args.project+'/results/'+args.name_dataset, args.name_dir_res)
    path_res = args.project+'/results/'+args.name_dataset+'/'+args.name_dir_res
    
    # metrics
    file_metrics = path_res+'/predict_metrics.csv'
    
    # feature,label
    config = Configuration(args.n_features, args.n_labels)

    # models 
    path_models = args.project+'/surrogate_models'
    path_model1 = path_models+'/test-'+args.name_dataset+'-obj1.sav' # f1
    path_model2 = path_models+'/test-'+args.name_dataset+'-obj2.sav' # lms
    
    problem = MLProblem(n_gene,
                        args.name_dataset,
                        config, 
                        n_threads, 
                        path_model1,
                        path_model2,
                        log_file,
                        file_metrics)
    
    algorithm = NSGA2(
        pop_size=args.len_population,
        sampling=MLSampling(config, n_gene, args.is_dataset_sparse),
        crossover=UniformCrossover(prob=0.5),
        mutation=MLMutation(0.05, config),
        eliminate_duplicates=NoDuplicateElimination() 
    )
    
    # Terminação: número máximo de gerações ou tolerância de 'tol' por 'period' gerações
    termination = TerminateIfAny(MaximumGenerationTermination(args.number_generation), RobustTermination(MultiObjectiveSpaceTermination(tol=termination_tol, n_skip=0), period=termination_period))
    
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=True,
        verbose=True
    )    
    
    # Resultados --------------------------------------------------------------
    
    HistoryRun.to_file(args.name_dataset, file_metrics)
    
    # Gráfico - espaço de objetivos
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', '-F1', 'Model Size', path_res+'/ObjectiveSpace.png')
    
    # Gráfico - hipervolume
    ref_point = np.array([0, 1e9])
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res, ref_point)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', path_res+'/Hypervolume.png')    
    
    # Prepara resultados para salvar em arquivo
    output_data = f'Execution time:{res.exec_time}\n'
      
    output_data += f'Best solution found:\n'
    for individual in res.X:
        output_data += f'{individual}\n'
     
    output_data += 'Classifiers:\n'
    for individual in res.X:
        is_normalize, fp_command, meka_command, weka_command = IndividualUtils.get_commands(config, individual)
        output_data += f"Normalize:{is_normalize}\n{fp_command}\n{meka_command}\n{weka_command}\n"
        
    output_data += f"Function value:\n{res.F}\n"
    
    list_f1 = [] # -f1
    list_f2 = [] # model size
    for l in res.F:
        list_f1 = np.append(list_f1, l[0])
        list_f2= np.append(list_f2, l[1])
    
    # f1
    output_data += 'F1 (macro averaged by label)\n['
    for i in range(len(list_f1)-1):
        output_data += str(-list_f1[i])+','
    output_data += str(-list_f1[len(list_f1)-1])+']\n'
     
    # size
    output_data += 'Model size\n['
    for i in range(len(list_f2)-1):
        output_data += str(list_f2[i])+','
    output_data += str(list_f2[len(list_f2)-1])+']\n'
    
    # Evaluation
    output_data += 'Evaluation:\n['
    for i in range(len(n_evals)-1):
        output_data += str(n_evals[i])+','
    output_data += str(n_evals[len(n_evals)-1])+']\n'
    
    # Hypervolume
    output_data += 'Hypervolume:\n['
    for i in range(len(hv)-1):
        output_data += str(hv[i])+','
    output_data += str(hv[len(hv)-1])+']\n'
    
    # History
    output_data += 'History:\n'
    for array in hist_F:
        if len(array) == 1:
            output_data += str(array[0][0])+', '+str(array[0][len(array[0])-1])+'\n'
        else:
            for i in range(len(array)-1):
                output_data += str(array[i][0])+', '+str(array[i][len(array[i])-1])+', '
            output_data += str(array[i+1][0])+', '+str(array[i+1][len(array[i+1])-1])+'\n'
    
    # Salva resultados
    output_path = pathlib.Path(f'{path_res}/results.txt')
    output_path.write_text(output_data)