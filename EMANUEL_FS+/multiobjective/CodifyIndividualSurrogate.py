import re

import inspect

import numpy as np

import shlex

import pandas as pd

from configuration.Configuration import Configuration


def get_all_indexes(config):
    dict_indexes = {}
    
    list_indexes_cols = np.array(['normalize'])
    
    dict_indexes['normalize'] = 0
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_fs_config(), 1, list_indexes_cols, dict_indexes,is_fs=True)
    #index, list_indexes_cols, dict_indexes = get_indexes(config.get_ml_ensemble_config(), index, list_indexes_cols, dict_indexes)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_ml_config(), index, list_indexes_cols, dict_indexes)
    #index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_ensemble_config(), index, list_indexes_cols, dict_indexes)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_config(), index, list_indexes_cols, dict_indexes, is_scl=True)
    index, list_indexes_cols, dict_indexes = get_indexes(config.get_sl_kernel_config(), index, list_indexes_cols, dict_indexes)
    
    return list_indexes_cols, dict_indexes


# Obtém os índices do dataframe/dataset para configurações ensemble, SLC e MLC
def get_indexes(config, i, list_indexes, dict_indexes, is_scl=False, is_fs=False):
    for algorithm, hyperparameters in config.items():
        #algorithm = algorithm.replace(without_pattern, '')
        list_indexes = np.append(list_indexes, algorithm)
        dict_indexes[algorithm] = i
        i = i + 1

        for hyp, values in hyperparameters.items():
            if hyp == '-normalize':
                continue
            elif hyp == 'if':
                function = values
                funcString = str(inspect.getsourcelines(function)[0])
                funcString = funcString.split('\'if\'')[1].split('if')
                blockThen = funcString[0]
                blockElse = funcString[1].split('else')[1]
                list_function = np.append(re.findall('-\D', blockThen), re.findall('-\D', blockElse))
                list_function = np.unique(list_function)
                for h in list_function: # para cada hiperparâmetro da função
                    list_indexes = np.append(list_indexes, algorithm+h)
                    dict_indexes[algorithm+h] = i
                    i = i + 1
            elif hyp != '-W' or is_scl == True:
                if is_fs == False:
                    list_indexes = np.append(list_indexes, algorithm+hyp)
                    dict_indexes[algorithm+hyp] = i
                else:
                    list_indexes = np.append(list_indexes, algorithm+'-'+hyp)
                    dict_indexes[algorithm+'-'+hyp] = i
                
                i = i + 1

    return i, list_indexes, dict_indexes



# verifica se uma string representa um número
def isNumber(n):
  try:
      float(n)
  except ValueError:
      return False
  return True


def codify(norm, fs_cmd, meka_cmd, weka_cmd, dict_indexes, config):
    ind = codify_feature_selection(norm, fs_cmd, dict_indexes, config)
    ind = codify_classifier(norm, meka_cmd, weka_cmd, dict_indexes, config, ind)
    return ind


def codify_feature_selection(norm, fs_cmd, dict_indexes, config):
    # cria vetor do individuo
    individual = np.full(len(dict_indexes), np.nan) # -1

    individual[0] = 0 if norm == False else 1

    fs_cmd = fs_cmd.replace('"', '')
    fs_cmd = fs_cmd.split('(', 1)
    algorithm = fs_cmd[0]
    fs_cmd = fs_cmd[1].rsplit(')', 1)[0].split(',')
    index_cmd = 0

    # seta algoritmo
    index_individual = dict_indexes.get(algorithm)
    individual[index_individual] = 0

    config_fs = config.get_fs_config() 
    config_algorithm = config_fs.get(algorithm)

    # seta hiperparâmetros
    for key, values in config_algorithm.items():
        val_hip = fs_cmd[index_cmd] # formato: hip=value
        index_cmd += 1

        val_hip = val_hip.split('=')[1]

        if isNumber(val_hip):
            val_hip = float(val_hip)

        index_hip = np.where(values == val_hip)[0][0]
        index_individual = dict_indexes.get(algorithm+'-'+key)
        individual[index_individual] = index_hip

    return individual


# codifica comandos meka e weka em um indivíduo estendido
def codify_classifier(norm, meka_cmd, weka_cmd, dict_indexes, config, individual):
    # cria vetor do individuo
    #individual = np.full(len(dict_indexes), np.nan) # -1
    
    is_pt = False
    
    #individual[0] = 0 if norm == False else 1
    
    # converte comandos meka e weka em arrays
    meka_cmd = shlex.split(meka_cmd)
    
    if not pd.isna(weka_cmd):
        weka_cmd = shlex.split(weka_cmd) 
    
    # obtém o algoritmo de classificação multirrótulo
    algorithm = meka_cmd[0]
    index_cmd = 1
    
    # MEKA - MULAN ---------------------------------------------------------------------   
    if 'MULAN' in algorithm:
        # Formato: meka.classifiers.multilabel.MULAN -S MLkNN.36
        meka_cmd = meka_cmd[2].split('.') # [MLkNN, 36]
        algorithm += '.'+meka_cmd[0] # meka.classifiers.multilabel.MULAN.MLkNN
        index_mulan = 1

        # seta como 1 o indice do indivíduo referente ao algoritmo
        index_individual = dict_indexes.get(algorithm)
        individual[index_individual] = 0 # 1

        config_ml = config.get_ml_config() 
        config_algorithm = config_ml.get(algorithm)

        for key, values in config_algorithm.items():
            if key == '-normalize':
                # ignora normalização
                continue

            else:
                # seta o indice do indivíduo com o índice do valor do hiperparâmetro
                val_hip = meka_cmd[index_mulan]
                index_mulan += 1

                if isNumber(val_hip):
                    val_hip = float(val_hip)

                index_hip = np.where(values == val_hip)[0][0]

                index_individual = dict_indexes.get(algorithm+key)
                individual[index_individual] = index_hip

    # MEKA ---------------------------------------------------------------------  
    else:  
        # seta como 1 o indice do indivíduo referente ao algoritmo
        index_individual = dict_indexes.get(algorithm)
        individual[index_individual] = 0 # 1

        config_ml = config.get_ml_config() 
        config_algorithm = config_ml.get(algorithm)
     
        params = {}
        for key, values in config_algorithm.items():
            if key == '-normalize':
                # ignora normalização
                continue

            elif key == '-W': 
                # o algoritmo multirrótulo é da abordagem transformação de problemas
                is_pt = True
                # obtém as configurações dos algoritmos SLCs
                config_slc = values

            elif key == 'if':
                function = config_algorithm[key]
                return_function = function(params)
                
                if isinstance(return_function, dict):
                    for key_dict, values_dict in return_function.items():
                        # obtém índice do indivíduo
                        index_individual = dict_indexes.get(algorithm+key_dict)
                        index_cmd, individual, params = get_one_cod(key_dict, values_dict, meka_cmd, index_cmd, individual, index_individual, params) 
            
            else:
                # seta o indice do indivíduo com o valor do hiperparâmetro
                index_individual = dict_indexes.get(algorithm+key)
                index_cmd, individual, params = get_one_cod(key, values, meka_cmd, index_cmd, individual, index_individual, params)            
        
        # SLC -----------------------------------------------------
        if is_pt == True:
            # obtém algoritmo de classificação monorrótulo (ensemble de SLC ou SLC)
            algorithm = weka_cmd[0]
            index_cmd = 1
            # --
            index_cmd = index_cmd + 1

            # obtém a configuração do algoritmo de classificação monorrótulo
            config_algorithm = config_slc.get(algorithm)

            # seta como 1 o indice do indivíduo referente ao algoritmo
            index_individual = dict_indexes.get(algorithm)
            individual[index_individual] = 0 # 1          

            params = {}
            for key, values in config_algorithm.items():
                if key == '-normalize':
                    # ignora normalização
                    continue

                elif key == 'if':
                    function = config_algorithm[key]
                    return_function = function(params)
                    
                    if isinstance(return_function, dict):
                        for key_dict, values_dict in return_function.items():
                            # obtém índice do indivíduo
                            index_individual = dict_indexes.get(algorithm+key_dict)
                            index_cmd, individual, params = get_one_cod(key_dict, values_dict, weka_cmd, index_cmd, individual, index_individual, params) 
                
                elif key == '-K' and isinstance(values, dict): # kernel
                    # seta smo-K com 1
                    index_individual = dict_indexes.get(algorithm+key)
                    individual[index_individual] = 0 # 1
    
                    index_cmd = index_cmd + 1 # -K

                    # obtém a configuração de hiperparâmetros do kernel - string
                    hips_kernel = weka_cmd[index_cmd]
                    hips_kernel = shlex.split(hips_kernel)

                    # obtém o kernel
                    index_hips_kernel = 0
                    kernel = hips_kernel[index_hips_kernel]
                    index_hips_kernel = index_hips_kernel + 1
                    
                    # seta como 1 o índice do indivíduo referente ao kernel
                    index_individual = dict_indexes.get(kernel) 
                    individual[index_individual] = 0 # 1
                    
                    # obtém a configuração do kernel                
                    config_kernel = values.get(kernel)
                    
                    for key_kernel, values_kernel in config_kernel.items():
                        # obtém índice do indivíduo
                        index_individual = dict_indexes.get(kernel+key_kernel)
                        index_hips_kernel, individual, params = get_one_cod(key_kernel, values_kernel, hips_kernel, index_hips_kernel, individual, index_individual, params) 
                        
                else:
                    # seta o indice do indivíduo com o valor do hiperparâmetro
                    index_individual = dict_indexes.get(algorithm+key)
                    index_cmd, individual, params = get_one_cod(key, values, weka_cmd, index_cmd, individual, index_individual, params)            
                               
    return individual


def get_one_cod(key, list_values, command, index_cmd, individual, index_individual, params):
    if list_values.dtype == bool: # list of bool
    
        #if key in command, então key é True           
        if index_cmd < len(command) and key == command[index_cmd]:
            val_hip = True
            index_cmd = index_cmd + 1
        else:
            val_hip = False

        params[key] = val_hip
        index_hip = np.where(list_values == val_hip)[0][0]
        individual[index_individual] = index_hip

    else:
        # obtém hiperparâmetro e seu valor 
        key_hip = command[index_cmd]
        index_cmd = index_cmd + 1
        val_hip = command[index_cmd]
        index_cmd = index_cmd + 1
        
        if isNumber(val_hip):
            val_hip = float(val_hip)
            #
            params[key_hip] = val_hip
            index_hip = np.where(list_values == val_hip)[0][0]
            individual[index_individual] = index_hip

        # string
        else:
            # busca índice de string em list_values, alerando padrão de aspas
            if val_hip in list_values:
                index_hip = np.where(list_values == val_hip)[0][0]
                params[key_hip] = val_hip

            elif '\''+val_hip+'\'' in list_values:
                index_hip = np.where(list_values == '\''+val_hip+'\'')[0][0]
                params[key_hip] = '\''+val_hip+'\''

            individual[index_individual] = index_hip
    
    return index_cmd, individual, params