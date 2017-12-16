import os
import numpy as np

inf_suffix = '.txt'


# load a instance from a file
def load_instance(fold_path, file_name):
    (l, _) = file_name.split('_')
    instance = []
    with open(os.path.join(fold_path, file_name), 'r') as inf:
        for line in inf:
            instance += [int(i) for i in line.strip()]
    return instance, int(l)


# load sample set according to a fold path
def load_sample_set(fold_path):
    instances = []
    ls = []
    for file_name in os.listdir(fold_path):
        if file_name.endswith(inf_suffix):
            instance, l = load_instance(fold_path, file_name)
            instances.append(instance)
            ls.append(l)
    return np.array(instances), np.array(ls).transpose()


