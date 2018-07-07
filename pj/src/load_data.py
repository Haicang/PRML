#! /usr/bin/python3

"""
Functions to load data
"""
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def load_car(filename='../dataset/car/car-vgood.dat'):
    names = ['Buying', 'Maint', 'Doors', 'Persons',
                       'Lug_boot', 'Safety']
    attributes = {'Buying':   ['low', 'med', 'high', 'vhigh'], 
                  'Maint':    ['low', 'med', 'high', 'vhigh'],
                  'Doors':    ['2', '3', '4', '5more'],
                  'Persons':  ['2', '4', 'more'],
                  'Lug_boot': ['small', 'med', 'big'],
                  'Safety':   ['low', 'med', 'high']}

    data = pd.read_csv(filename, skiprows=11, 
                       names= names + ['Class'],
                       skipinitialspace=True,
                       true_values=['positive', ], false_values=['negative', ])

    # convert labels to digits
    for name in names:
        data[name] = pd.Categorical(data[name], ordered=True, categories=attributes[name])
        data[name] = data[name].cat.codes

    return shuffle_split(data)


def load_wisconsin(filename='../dataset/wisconsin/wisconsin.dat'):
    data = pd.read_csv(filename, skiprows=14,
                       names=['ClumpThickness', 'CellSize', 'CellShape',
                        'MarginalAdhesion', 'EpithelialSize', 'BareNuclei',
                        'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class'],
                        skipinitialspace=True, 
                        true_values=['positive',], false_values=['negative',])
    return shuffle_split(data)


def load_yeast(filename='../dataset/yeast/yeast-2_vs_4.dat'):
    data = pd.read_csv(filename, skiprows=13,
                       names=['Mcg', 'Gvh', 'Alm', 'Mit', 'Erl', 'Pox', 
                       'Vac', 'Nuc', 'Class'],
                       skipinitialspace=True,
                       true_values=['positive',], false_values=['negative',])
    return shuffle_split(data)


def shuffle_split(data):
    """
    Helper function to convert pd.DataFrame to
    features, and targets in np.ndarray with proper dtype
    """
    data = shuffle(data, random_state=1)
    features = np.array(data.values[:, :-1], dtype=np.double)
    targets = np.array(data.values[:, -1], dtype=np.int32)
    return (features, targets)


# The following functions are specific to pre-split data
def load_5_fold(dataset_name):
    names = ('car', 'wisconsin', 'yeast')
    assert dataset_name in names
    fns = {'car': load_car,
            'wisconsin': load_wisconsin,
            'yeast': load_yeast}
    dirnames = {'car': 'car-vgood-5-fold',
                 'wisconsin': 'wisconsin-5-fold',
                 'yeast': 'yeast-2_vs_4-5-fold'}
    filenames = {'car': 'car-vgood-5-{}{}.dat',
                 'wisconsin': 'wisconsin-5-{}{}.dat',
                 'yeast': 'yeast-2_vs_4-5-{}{}.dat'}

    X, y = [], []
    for i in range(1, 6):
        path_train = '../dataset/' + dataset_name + '/' \
            + dirnames[dataset_name] + '/' \
            + filenames[dataset_name].format(i, 'tra')
        path_test = '../dataset/' + dataset_name + '/' \
            + dirnames[dataset_name] + '/' \
            + filenames[dataset_name].format(i, 'tst')
        
        train = fns[dataset_name](path_train)
        test = fns[dataset_name](path_test)

        X.append((train[0], test[0]))
        y.append((train[1], test[1]))

    return X, y
