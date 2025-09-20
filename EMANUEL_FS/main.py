import os
import sys
import pathlib
import tarfile
import shutil

import argparse

import numpy as np

from configuration.Configuration import Configuration

from multiobjective.MLMutation import MLMutation
from multiobjective.MLProblem import MLProblem
from multiobjective.MLSampling import MLSampling
from multiobjective.IndividualUtils import IndividualUtils

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.duplicate import NoDuplicateElimination

from pymoo.core.termination import TerminateIfAny
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_gen import MaximumGenerationTermination

from utils.Graphic import Graphic
from utils.ManipulateHistory import ManipulateHistory


# -------------------------------------------------------------------
def create_dir(destination, name_dir):    
    cmd = 'if [ ! -d '+destination+'/'+name_dir+' ]\nthen\n\tmkdir '+destination+'/'+name_dir+'\nfi'
    os.system(cmd)

def compress_file(file, source_folder):
   with tarfile.open(file, "w:gz") as tar:
        tar.add(source_folder, arcname=os.path.basename(source_folder))  

def remove_file(path_file):
    os.unlink(path_file)

def remove_dir(path_dir):
    shutil.rmtree(path_dir)

# -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    '''parser.add_argument('--run', type=int, required=False, default=1)
    parser.add_argument('--n_labels', type=int, required=False, default=6)
    parser.add_argument('--n_features', type=int, required=False, default=72)
    parser.add_argument('--is_dataset_sparse', action='store_true', required=False, default=False) # para true utiliza --is_dataset_sparse
    parser.add_argument('--name_dataset', type=str, required=False, default='emotions')
    parser.add_argument('--meka_classpath', type=str, required=False, default='/home/u798230/Experimento4/emanuel/meu-meka-1.9.8/lib/')
    parser.add_argument('--java_command', type=str, required=False, default='/home/u798230/miniconda3/envs/AmbientePyIT/bin/java')
    parser.add_argument('--project', type=str, required=False, default= '/home/u798230/Experimento4/emanuel')
    parser.add_argument('--k_folds', type=int, required=False, default=3)
    parser.add_argument('--n_threads', type=int, required=False, default=3)
    parser.add_argument('--fp_limit_time', type=int, required=False, default=30)
    parser.add_argument('--mlc_limit_time', type=int, required=False, default=60)
    parser.add_argument('--len_population', type=int, required=False, default=3)
    parser.add_argument('--number_generation', type=int, required=False, default=1)'''

    parser.add_argument('--run', type=int, required=False, default=1)
    parser.add_argument('--n_labels', type=int, required=False, default=7)
    parser.add_argument('--n_features', type=int, required=False, default=19)
    parser.add_argument('--is_dataset_sparse', action='store_true', required=False, default=False) # para true utiliza --is_dataset_sparse
    parser.add_argument('--name_dataset', type=str, required=False, default='flags')
    parser.add_argument('--meka_classpath', type=str, required=False, default='./meu-meka-1.9.8/lib/')
    parser.add_argument('--java_command', type=str, required=False, default='/home/dellvale/miniconda3/envs/AmbienteMEKA/bin/java')
    parser.add_argument('--project', type=str, required=False, default= '/media/dellvale/Dados/Doutorado/Experimentos/Experimento4/emanuel')
    parser.add_argument('--k_folds', type=int, required=False, default=3)
    parser.add_argument('--n_threads', type=int, required=False, default=3)
    parser.add_argument('--fp_limit_time', type=int, required=False, default=30)
    parser.add_argument('--mlc_limit_time', type=int, required=False, default=40)
    parser.add_argument('--len_population', type=int, required=False, default=10)
    parser.add_argument('--number_generation', type=int, required=False, default=10)

    args = parser.parse_args() 

    n_gene = 40 # do indivíduo
    termination_period = 10
    termination_tol = 0.001 # de melhora

    # Ambiente -----------------------------------------------------------------

    path_dataset = args.project+'/datasets'

    create_dir(args.project+'/results', args.name_dataset)
    create_dir(args.project+'/results/'+args.name_dataset, args.name_dataset+str(args.run))
    path_results = args.project+'/results/'+args.name_dataset+'/'+args.name_dataset+str(args.run)

    # AutoML -------------------------------------------------------------------

    config = Configuration(args.n_features, args.n_labels)
    problem = MLProblem(n_gene,
                        args.k_folds,
                        args.n_threads,
                        args.fp_limit_time, 
                        args.mlc_limit_time,
                        config, 
                        args.java_command, 
                        args.meka_classpath, 
                        args.n_labels, 
                        args.n_features,
                        args.is_dataset_sparse,
                        args.name_dataset, 
                        path_dataset,
                        path_results+'/metrics.csv'
    )
    
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
    
    # Results -------------------------------------------------------------------

    # Gráfico - espaço de objetivos
    Graphic.plot_scatter(res.F[:, 0], res.F[:, 1], 'Objective Space', '-F1', 'Model Size', path_results+'/ObjectiveSpace.png')
    
    # Gráfico - hipervolume
    ref_point = np.array([0, 1e9])
    n_evals, hist_F, hv = ManipulateHistory.get_hypervolume(res, ref_point)
    Graphic.plot_graphic(n_evals, hv, 'Convergence-Hypervolume', 'Evaluations', 'Hypervolume', path_results+'/Hypervolume.png')    
    
    # Prepara resultados para salvar em arquivo
    output_data = f'Number of queries on the classifier and objectives map: {problem.n_queries}\n'
    output_data += f'Number of feature preprocessing algorithms exceeding the time limit: {problem.n_fp_timeouts}\n'
    output_data += f'Number of classifiers exceeding the time limit: {problem.n_mlc_timeouts}\n'
    output_data += f'Number FP exception: {problem.n_fp_exceptions}\n'
    output_data += f'Number exception: {problem.n_mlc_exceptions}\n'
    output_data += f'Execution time:{res.exec_time}\n'

    output_data += 'Best solution found:\n'
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
     
    # model size
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
    output_path = pathlib.Path(f'{path_results}/results.txt')
    output_path.write_text(output_data)
    
    compress_file(args.project+'/results/'+args.name_dataset+'/'+args.name_dataset+str(args.run)+'.tar.gz', path_results) # +'/'
    remove_dir(path_results)    
