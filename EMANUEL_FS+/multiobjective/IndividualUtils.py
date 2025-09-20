import numpy as np

class IndividualUtils:

    # In decoding MLC algorithms:
    #   Boolean hyp.s may or may not appear in the command, but they occupy positions in the individual
    #   Hyp.s dependent on conditionals are counted in the individual depending on the condition

    # -----------------------------------------------------------------------------
    # Decodifica o indivíduo em comandos feature preprocessing, meka e weka
    # 0 = normalize
    # 1 = feature preprocessing
    # hyperparameters os feature preprocessing
    # MLC algorithm
    # hyperpameters os MLC algorithm
    # -----------------------------------------------------------------------------
    def get_commands(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0
        # normalization, depends on the MLC or SLC algorithm
        
        # index 1 - FP
        index_algorithm = individual[1] 
        algorithm = config.get_fs_algorithms()[index_algorithm%len(config.get_fs_algorithms())]

        i = 2
        #if 'NO_FEATURE_PREPROCESSING' in algorithm:
        #    fp_command = algorithm
        #else: 
        config_fp = config.get_fs_config()
        config_algorithm = config_fp.get(algorithm)
        i, fp_command = IndividualUtils.getFPCommand(individual, algorithm, config_algorithm, i)

        # other indexes
        # MLC algorithm
        index_algorithm = individual[i] 
        algorithm = config.get_ml_algorithms()[index_algorithm%len(config.get_ml_algorithms())]
        meka_command = algorithm
        weka_command = None
        i += 1
        
        # MULAN algorithms 
        if 'MULAN' in algorithm:
            config_ml = config.get_ml_config() 
            config_algorithm = config_ml.get(algorithm)
            i, is_normalize, meka_command = IndividualUtils.command_mulan(individual, algorithm, config_algorithm, i)
            
        # MEKA algorithms
        else:
            # ensemble
            if 'meta' in algorithm:
                is_mlc_ensemble = True
                config_ml = config.get_ml_ensemble_config()
                config_algorithm = config_ml.get(algorithm)
                
                i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
                meka_command = meka_command + command + ' -W ' + algorithm + ' --'
                config_algorithm = config_algorithm['-W'][algorithm]
                
            # MLC           
            if is_mlc_ensemble == False:
                config_ml = config.get_ml_config() 
                config_algorithm = config_ml.get(algorithm)
            
            if '-W' in config_algorithm.keys():
                is_pt = True
            
            i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
            meka_command = meka_command + command
            
            # SLC ensemble
            if is_pt == True:
                weka_command = algorithm + ' --'
                config_algorithm = config_algorithm['-W'][algorithm]
                
                # ensemble
                if 'meta' in algorithm or 'LWL' in algorithm:
                    i, is_normalize, command, algorithm = IndividualUtils.command_aux(individual, config_algorithm, i)
                    weka_command = weka_command + command + ' -W ' + algorithm + ' --'
                    config_algorithm = config_algorithm['-W'][algorithm]
            
                # SLC
                i, is_normalize, command = IndividualUtils.command_slc_aux(individual, config, config_algorithm, i)
                weka_command = weka_command + command

        # correção de aspas
        if weka_command is not None:
            if 'weka.classifiers.bayes.BayesNet' in weka_command:
                weka_command = IndividualUtils.set_bayes_net(weka_command) 
                #print(f'--- {weka_command} ---')
                
        #return i, is_normalize, fp_command, meka_command, weka_command    # para verificar o tamanho do indivíduo       
        return is_normalize, fp_command, meka_command, weka_command         
    

    # correção de aspas
    def set_bayes_net(weka_cmd):

        if '-Q weka.classifiers.bayes.net.search.local.K2 -- -P 1' in weka_cmd:
            weka_cmd = weka_cmd.replace('-Q weka.classifiers.bayes.net.search.local.K2 -- -P 1', '-Q \'weka.classifiers.bayes.net.search.local.K2 -- -P 1\'')
        
        if '-Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 1' in weka_cmd:
            weka_cmd = weka_cmd.replace('-Q weka.classifiers.bayes.net.search.local.HillClimber -- -P 1', '-Q \'weka.classifiers.bayes.net.search.local.HillClimber -- -P 1\'')
        
        if '-Q weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1' in weka_cmd:
            weka_cmd = weka_cmd.replace('-Q weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1', '-Q \'weka.classifiers.bayes.net.search.local.LAGDHillClimber -- -P 1\'')
        
        if '-Q weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1' in weka_cmd:
            weka_cmd = weka_cmd.replace('-Q weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1', '-Q \'weka.classifiers.bayes.net.search.local.TabuSearch -- -P 1\'')
        
        return weka_cmd
    # -----------------------------------------------------------------------------


    # -----------------------------------------------------------------------------
    # The function decodes feature preprocessing algorithm into a Python algorithm
    # -----------------------------------------------------------------------------
    def getFPCommand(individual, algorithm, config_algorithm, i):
        is_first = True
    
        command = algorithm+'('
        for variable, all_values in config_algorithm.items():
            if is_first == False:
                command += ','

            value = all_values[individual[i]%len(all_values)]
            command += variable+'='+str(value)
            is_first = False
            i += 1

        command += ')'

        return i, command
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Decoding function for: MULAN
    # -------------------------------------------------------------------------
    def command_mulan(individual, algorithm, config_algorithm, i):
        # example of algorithm: meka.classifiers.multilabel.MULAN.MLkNN
        mlc = algorithm.split('.')[-1]
        meka_command = 'meka.classifiers.multilabel.MULAN -S ' + mlc

        is_normalize = False

        for var, all_values in config_algorithm.items():

            if var == '-normalize':
                value = all_values[individual[0]%len(all_values)]
                is_normalize = False if value%2 == 0 else True
                # does not increase i
            else:
                value = all_values[individual[i]%len(all_values)]
                meka_command += '.'+str(value)
                i += 1
        
        return i, is_normalize, meka_command
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # Decoding function for: MLC ensemble, MLC and SLC ensemble
    # -------------------------------------------------------------------------
    def command_aux(individual, config_algorithm, i):
        command = ''
        params = {}
        algorithm = ''
        is_normalize = None
        for variable in config_algorithm.keys():
            if variable == 'if': # function
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, all_values in dictionary.items():
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                                command = command + ' ' + var + ' ' + str(value)
                        
                        i = i + 1  
            else:
                all_values = config_algorithm[variable]
                
                if variable == '-normalize':
                    value = all_values[individual[0]%len(all_values)]
                    is_normalize = False if value%2 == 0 else True
                    # does not increase i
                elif variable == '-W':
                    all_values = list(all_values.keys())
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value
                    i = i + 1
                    # print(all_values)
                else:
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                    
                    i = i + 1
            
        return i, is_normalize, command, algorithm
    # -------------------------------------------------------------------------
    

    # -------------------------------------------------------------------------
    # Decoding function for SLC algorithm
    # -------------------------------------------------------------------------
    def command_slc_aux(individual, config, config_algorithm, i):
        command = ''
        params = {}
        is_normalize = None
        for variable in config_algorithm.keys():
            if variable == 'if': # function
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, all_values in dictionary.items():
                        value = all_values[individual[i]%len(all_values)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                                
                        i = i + 1    
            else: # list or kernel
                all_values = config_algorithm[variable]
            
                if variable == '-normalize':
                    value = all_values[individual[0]%len(all_values)]
                    is_normalize = False if value%2 == 0 else True
                    # does not increase i
                elif isinstance(all_values, np.ndarray): # list
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    
                    if isinstance(value, np.bool_):
                        if value==True:
                            command = command + ' ' + variable
                    else:
                        command = command + ' ' + variable + ' ' + str(value)
                        
                    i = i + 1
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       

                    command = command + ' ' + variable + ' \"' + kernel
                    i = i + 1
                
                    for var, all_values_kernels in config_kernel.items():
                        value = all_values_kernels[individual[i]%len(all_values_kernels)]
                        params[var] = value
                        
                        if isinstance(value, np.bool_):
                            if value==True:
                                command = command + ' ' + var
                        else:
                            command = command + ' ' + var + ' ' + str(value)
                            
                        i = i + 1
                              
                    command = command + '\"'
                    
        return i, is_normalize, command
    # -------------------------------------------------------------------------
    

    # -------------------------------------------------------------------------
    # Get the number of genes that represent the individual
    # This function is used on mutation
    # -------------------------------------------------------------------------
    def get_lenght_individual(config, individual):
        is_pt = False # PT - problem transformation
        is_mlc_ensemble = False
        
        # index 0
        # normalization, depends on the MLC or SLC algorithm
        
        # index 1 - FP
        index_algorithm = individual[1] 
        algorithm = config.get_fs_algorithms()[index_algorithm%len(config.get_fs_algorithms())]

        i = 2
        if 'NO_FEATURE_PREPROCESSING' not in algorithm:
            config_fp = config.get_fs_config()
            config_algorithm = config_fp.get(algorithm)
            i += len(config_algorithm)

        # other indexes
        # MLC algorithm
        index_algorithm = individual[i] 
        algorithm = config.get_ml_algorithms()[index_algorithm%len(config.get_ml_algorithms())]
        i += 1
        
        # ensemble
        if 'meta' in algorithm:
            is_mlc_ensemble = True
            config_ml = config.get_ml_ensemble_config()
            config_algorithm = config_ml.get(algorithm)
            
            i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
            config_algorithm = config_algorithm['-W'][algorithm]
            
        # MLC           
        if is_mlc_ensemble == False:
            config_ml = config.get_ml_config()
            config_algorithm = config_ml.get(algorithm)
        
        if '-W' in config_algorithm.keys():
            is_pt = True
        
        i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
        
        # SLC ensemble
        if is_pt == True:
            config_algorithm = config_algorithm['-W'][algorithm]
            
            # ensemble
            if 'meta' in algorithm or 'LWL' in algorithm:
                i, algorithm = IndividualUtils.get_lenght_aux(individual, config_algorithm, i)
                config_algorithm = config_algorithm['-W'][algorithm]
        
            # SLC
            i = IndividualUtils.get_lenght_slc_aux(individual, config, config_algorithm, i)
            
        return i # tamanho
    # -----------------------------------------------------------------------------


    # -----------------------------------------------------------------------------
    # Get the number of genes occupied by: MLC ensemble, MLC and SLC ensemble
    # -----------------------------------------------------------------------------
    def get_lenght_aux(individual, config_algorithm, i):
        params = {}
        algorithm = ''
        for variable, all_values in config_algorithm.items():
            if variable == 'if': # função
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1  
            else:
                if variable == '-normalize':
                    continue
                    # does not count in size
                elif variable == '-W': # -W (algorithm)
                    all_values = list(all_values.keys())
                    value = all_values[individual[i]%len(all_values)]
                    algorithm = value
                    i = i + 1
                else: 
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value
                    i = i + 1
            
        return i, algorithm
    # -----------------------------------------------------------------------------


    # -----------------------------------------------------------------------------    
    # Get the number of genes occupied by SLCs algorithms
    # -----------------------------------------------------------------------------
    def get_lenght_slc_aux(individual, config, config_algorithm, i):
        params = {}
        for variable, all_values in config_algorithm.items():
            if variable == 'if': # function
                function = config_algorithm[variable]
                return_function = function(params)
        
                if isinstance(return_function, dict):
                    dictionary = return_function
                    for var, values in dictionary.keys():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1    
            else: # list or kernel
            
                if variable == '-normalize':
                    continue
                    # does not count in size 
                elif isinstance(all_values, np.ndarray): # list
                    value = all_values[individual[i]%len(all_values)]
                    params[variable] = value  
                    i = i + 1
                    
                else: # kernel
                    kernels = list(all_values.keys()) 
                    kernel = kernels[individual[i]%len(kernels)]
                    config_kernel = config.get_sl_kernel_config().get(kernel)                       
                    i = i + 1
                
                    for var, values in config_kernel.items():
                        value = values[individual[i]%len(values)]
                        params[var] = value
                        i = i + 1
                              
        return i
    # -----------------------------------------------------------------------------
