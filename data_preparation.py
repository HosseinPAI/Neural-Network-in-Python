# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:50:19 2020

@author: MrHossein
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def prepare_data(input_data, normalization=True, n_feature_output=128, pca=True):
    train_data_in = np.loadtxt(input_data[0], dtype=np.float32, delimiter=',')
    train_label_in = np.loadtxt(input_data[1], dtype=np.int32, delimiter=',')
    test_data = np.loadtxt(input_data[2], dtype=np.float32, delimiter=',')
    test_label = np.loadtxt(input_data[3], dtype=np.int32, delimiter=',')

    if pca == True:
        total_data = np.concatenate((train_data_in, test_data))
        myPCA = PCA(n_components=n_feature_output)
        data_reduce = myPCA.fit_transform(total_data)

        train_data_in = data_reduce[0:60000, :]
        test_data = data_reduce[60000:, :]

    train_data, valid_data, train_label, valid_label = train_test_split(train_data_in, train_label_in, test_size=0.1,
                                                                        random_state=130)

    if normalization == True:
        train_data = normalize(train_data, norm='max', axis=0)
        valid_data = normalize(valid_data, norm='max', axis=0)
        test_data = normalize(test_data, norm='max', axis=0)

    return train_data, train_label, valid_data, valid_label, test_data, test_label
