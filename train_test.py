# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:07:20 2020

@author: MrHossein
"""

from Models import NN_model
from Models import GRBF_Model
import data_preparation as dp
import matplotlib.pyplot as plt
import time


def train_test(params):
    start_time = time.time()
    # Read Data from CSV Files and Prepare Them for Impose on the Model
    train_data, train_label, valid_data, valid_label, test_data, \
        test_label = dp.prepare_data([params.dataset_dir+params.train_data, params.dataset_dir+params.train_label,
                                      params.dataset_dir+params.test_data, params.dataset_dir+params.test_label],
                                     normalization=params.normalization,
                                     n_feature_output=params.num_of_features,
                                     pca=params.pca)

    if params.model == 'NN':
        m = 'Simple Neural Network'
        print('\n     >>>>>> Simple Neural Network <<<<<<    ')
        nn_model = NN_model.Nueral_net(params.num_of_features, params.num_of_hidden_layer,
                                       params.hidden_layers, params.num_of_classes, std=params.std)
    else:
        m = 'Gaussian RBF Network'
        print('\n     >>>>>> Gaussian RBF Neural Network <<<<<<    ')
        nn_model = GRBF_Model.Gaussian_RBF_NN(params.num_of_features, params.num_of_hidden_layer,
                                              params.hidden_layers, params.num_of_classes, std=params.std)

    train_data_loss, valid_data_loss, train_data_acc, \
        valid_data_acc = nn_model.train_function(train_data.T, train_label.T, valid_data.T, valid_label.T,
                                                 learning_rate=params.learning_rate,
                                                 learning_rate_decay=params.learning_rate_decay,
                                                 momentum=params.momentum,
                                                 batch_size=params.batch_size, reg=params.reg,
                                                 num_of_iters=params.num_of_iters)

    # Impose Test Data and Show the Result
    print('===================================================================')
    print('The Acuuracy of Model on Test Data : {0:.2f}'.format(
        (nn_model.Test_function(test_data.T, test_label.T)) * 100), '%')
    print('===================================================================')

    end_time = time.time()
    print('\nTotal Time for training and testing model: {0:.2f}'.format(end_time - start_time), ' Sec')

    # Plot Data
    plt.figure(figsize=(8, 6), dpi=200)
    plt.title('Train Vs Validation Loss in {0}'.format(m), color='darkblue')
    plt.plot(train_data_loss, color='orange', label='Train data')
    plt.plot(valid_data_loss, color='blue', label='Valid data')
    plt.legend()
    plt.savefig(params.save_dir+'Loss_{0}.jpg'.format(params.model), dpi=200)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.title('Train Vs Validation Accuracy in {0}'.format(m), color='darkblue')
    plt.plot(train_data_acc, color='orange', label='Train Data')
    plt.plot(valid_data_acc, color='blue', label='Validation data')
    plt.legend()
    plt.savefig(params.save_dir+'Acc_{0}.jpg'.format(params.model), dpi=200)
