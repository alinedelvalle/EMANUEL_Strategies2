import arff

import tempfile

import numpy as np


def __read_dataset(n_labels, path_dataset):
        arff_frame = arff.load(
            open(path_dataset, 'r'), encode_nominal=False, return_type=arff.DENSE  
        )

        #relation = arff_frame['relation']
        attributes = arff_frame['attributes']
        values = np.array(arff_frame['data'], dtype=object)

        for i in range(len(attributes)):
            att_type = attributes[i][1]

            if att_type == 'NUMERIC':
                values[:,i] = values[:,i].astype(float)

            # categorical
            if isinstance (att_type, list):
                values[:,i] = values[:,i].astype(int)
            
        X = values[:, :-n_labels]
        y = values[:, -n_labels:]
        y = y.astype(int)

        return X, y  
    

def __save_new_arrf(n_labels, name_dataset, X, y, pfx, temp_datasets):
        relation = name_dataset+': -C -'+str(n_labels)

        attributes = []
        for i in range(X.shape[1]):
            if isinstance(X[0, i], float):
                attributes.append(('X'+str(i), 'NUMERIC')) 
            elif isinstance(X[0, i], int): 
                list_unique = [str(value) for value in np.unique(X[:, i])]
                attributes.append(('X'+str(i), list_unique)) 
        for i in range(y.shape[1]): attributes.append(('Y'+str(i), ['0', '1']))

        values = np.concatenate((X, y), axis=1, dtype=object)

        arff_frame = {}
        arff_frame['relation'] = relation
        arff_frame['attributes'] = attributes
        arff_frame['data'] = values

        arff_save = arff.dumps(arff_frame)

        temp_file = tempfile.NamedTemporaryFile(dir=temp_datasets, prefix=pfx, suffix='.arff', delete=False)
        with open(temp_file.name, 'w', encoding='utf8') as fp:
            fp.write(arff_save)

        return temp_file.name