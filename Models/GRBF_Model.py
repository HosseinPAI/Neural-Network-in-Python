"""
Created on Mon Mar 16 21:44:13 2020

@author: MrHossein
"""
import numpy as np
import math


class Gaussian_RBF_NN(object):

    # input_feature_size   : the Size of input features, for exmple, it is 784 for Fashion MNIST
    # num_of_hidden_layers : The Number of Hidden Layyers thet you want to make, for example = 3
    # size_of_each_layer   : The size of each hidden layer in neural net.It should be like a list : [12,15,10]
    # output_size          : The size of output classification. for example, it is 10 for Fashion MNIST

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def __init__(self, input_feature_size, num_of_hidden_layers, size_of_each_layer, output_size, std=0.001):

        self.total_layer = len(size_of_each_layer) + 2
        self.total_layer_size = [input_feature_size] + size_of_each_layer + [output_size]
        self.out_size = output_size
        self.num_of_hidden_layers = num_of_hidden_layers

        self.Bs = []
        self.dBs = []
        self.delta_Bs = []
        for i in range(self.total_layer - 1):
            self.Bs.append(np.zeros((self.total_layer_size[i + 1], 1)))
            self.dBs.append(np.zeros((self.total_layer_size[i + 1], 1)))
            self.delta_Bs.append(np.zeros((self.total_layer_size[i + 1], 1)))

        self.Ws = []
        self.dWs = []
        self.delta_Ws = []
        for i in range(self.total_layer - 1):
            self.Ws.append(std * np.random.randn(self.total_layer_size[i + 1], self.total_layer_size[i]))
            self.dWs.append(np.zeros((self.total_layer_size[i + 1], self.total_layer_size[i])))
            self.delta_Ws.append(np.zeros((self.total_layer_size[i + 1], self.total_layer_size[i])))

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def relu(self, X, derivative=False):
        if derivative:
            return 1.0 * (X > 0)
        return np.maximum(0, X)

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def gelu(self, X, derivative=False):
        if derivative:
            return 0.5 * np.tanh(0.0356774 * X * X * X + 0.797885 * X) + \
                (0.0535161 * X * X * X + 0.398942 * X) * (np.square(2.0 / (
                            (np.exp(0.0356774 * X * X * X + 0.797885 * X)) + (
                        np.exp(-0.0356774 * X * X * X - 0.797885 * X))))) + 0.5

        return 0.5 * X * (1.0 + np.tanh((np.sqrt(2.0 / np.pi)) * (X + 0.044715 * X * X * X)))

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def original_gelu(self, X, derivative=False):
        X = np.array(X)
        output = np.zeros(np.shape(X))
        if derivative == True:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    output[i][j] = 0.5 * (1 + math.erf(X[i][j] / math.sqrt(2))) + (X[i][j] / math.sqrt(2 * np.pi)) * (
                        np.exp(-(X[i][j] ** 2) / 2))
        else:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    output[i][j] = 0.5 * X[i][j] * (1 + math.erf(X[i][j] / math.sqrt(2)))

        return output

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def LG_function(self, X, true_out_index=None, reg=0, lamda=550, validatian=False):

        # ------------------------------------------------
        #                 Forward Pass
        # ------------------------------------------------
        Zs = []
        As = []
        As.append(X)
        for i in range(self.total_layer - 1):
            Zs.append(np.dot(self.Ws[i], As[i]) + self.Bs[i])
            if (i != (self.total_layer - 2)):
                As.append(self.gelu(Zs[i]))

        output = Zs[i]
        scores = np.abs(Zs[i])

        if true_out_index is None:
            return scores

        # ------------------------------------------------
        #          Calculate Metric Learning Loss
        # ------------------------------------------------
        true_out_index = np.array(true_out_index)
        # Claculate Loss Term1
        mask1 = np.zeros((self.out_size, len(true_out_index)))
        mask1[true_out_index, np.arange(0, len(true_out_index))] = 1
        True_distance = mask1 * scores
        loss_term1 = np.sum(True_distance) / X.shape[1]

        # Calculate Loss Term2
        Loss_matrix = lamda - scores
        Loss_matrix[true_out_index, np.arange(0, Loss_matrix.shape[1])] = 0
        Loss_matrix[Loss_matrix < 0] = 0
        loss_term2 = np.sum(Loss_matrix) / X.shape[1]

        # Calculate Regularization
        Reg_term = 0.5 * reg * np.sum(np.square(self.Ws[-1]))

        # Calculate Total Loss
        total_loss = loss_term2 + loss_term1 + Reg_term

        if validatian == True:
            return total_loss
        # ------------------------------------------------
        #           Calculate Gradian
        # ------------------------------------------------
        # Claculate Gradiant of Last Layer
        True_distance[True_distance > 0] = 1
        Loss_matrix[Loss_matrix > 0] = -1

        Loss_gradiant = True_distance + Loss_matrix
        output[output > 0] = 1
        output[output < 0] = -1
        absolute_prime = output
        Loss_gradiant = Loss_gradiant * absolute_prime
        self.dWs[-1] = (np.dot(Loss_gradiant, As[-1].T)) + reg * self.Ws[-1]
        self.dBs[-1] = (np.dot(Loss_gradiant, np.ones((X.shape[1], 1))))

        delta = Loss_gradiant
        for i in range(self.total_layer - 2):
            temp = Zs[-(i + 2)]
            f_prime = self.gelu(temp, derivative=True)
            delta = np.dot(self.Ws[-(i + 1)].T, delta) * f_prime
            self.dBs[-(i + 2)] = np.dot(delta, np.ones((X.shape[1], 1)))
            self.dWs[-(i + 2)] = np.dot(delta, As[-(i + 2)].T)

        return total_loss, self.dWs, self.dBs

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def train_function(self, X_train, y_train, X_valid, y_valid, learning_rate=1e-3, learning_rate_decay=0.95,
                       momentum=0.95, batch_size=128, reg=1e-5, num_of_iters=100):

        num_of_log_data = 100
        log_point = int(num_of_iters / num_of_log_data)
        if log_point == 0:
            log_point = 1

        num_of_train_data = X_train.shape[1]

        train_data_loss = []
        valid_data_loss = []
        train_data_acc = []
        valid_data_acc = []

        counter = 0
        cmp = num_of_iters / 50
        print('\nTraining Neural Network is started, Please wait !!!')
        # index = 0
        for i in range(num_of_iters):
            counter += 1
            if counter % cmp == 0:
                print('.', end='')

            batch_data = None
            batch_label = None

            # Choose random data from validation data to calculate Loss
            if i % log_point == 0:
                rand_mask = np.random.permutation(X_valid.shape[1])
                batch_data = X_valid[:, rand_mask[:batch_size]]
                batch_label = y_valid[rand_mask[:batch_size]]
                loss = self.LG_function(batch_data, true_out_index=batch_label, reg=reg, validatian=True)
                valid_data_loss.append(loss)

            # Choose random data from input based on batch size to train NN
            rand_mask = np.random.permutation(num_of_train_data)
            batch_data = X_train[:, rand_mask[:batch_size]]
            batch_label = y_train[rand_mask[:batch_size]]

            # calculate loss and gradients
            loss, dws, dbs = self.LG_function(batch_data, true_out_index=batch_label, reg=reg, validatian=False)
            if i % log_point == 0:
                train_data_loss.append(loss)

            for j in range(self.total_layer - 1):
                self.delta_Ws[j] = momentum * self.delta_Ws[j] - learning_rate * dws[j]
                self.Ws[j] += self.delta_Ws[j] - learning_rate * reg * self.Ws[j]

                self.delta_Bs[j] = momentum * self.delta_Bs[j] - learning_rate * dbs[j]
                self.Bs[j] += self.delta_Bs[j] - learning_rate * reg * self.Bs[j]

            if i % log_point == 0:
                # Check accuracy
                train_data_acc.append((self.predict(batch_data) == batch_label).mean())
                valid_data_acc.append((self.predict(X_valid) == y_valid).mean())
                learning_rate *= learning_rate_decay

            # if i % log_point == 0:
            # print('iteration %d / %d: loss = %f   Train Acc = %s    Valid Acc = %s' % (i, num_of_iters, loss, train_data_acc[index], valid_data_acc[index]))
            # index += 1

        print('\nTraining Neural Network is Finished. You can see the results. :)\n')
        return train_data_loss, valid_data_loss, train_data_acc, valid_data_acc

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def predict(self, X):
        y_prediction = None
        y_prediction = np.argmin(self.LG_function(X), axis=0)

        return y_prediction

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    def Test_function(self, X_test, y_test):
        return (self.predict(X_test) == y_test).mean()
