import os
import argparse


class parameters():
	def __init__(self):
		super(parameters, self).__init__()
		self.parser = argparse.ArgumentParser(description='A simple Neural Network Model with Python')

	def initial_params(self):
		self.parser.add_argument('--num_of_features', type=int, default=128, help='The dimension of input features')
		self.parser.add_argument('--num_of_classes', type=int, default=10, help='Number of output classes')
		self.parser.add_argument('--dataset_dir', default='', help='Dataset Directory')
		self.parser.add_argument('--train_data', type=str, default='trainData.csv', help='Train Data')
		self.parser.add_argument('--train_label', type=str, default='trainLabels.csv', help='Train label')
		self.parser.add_argument('--test_data', type=str, default='testData.csv', help='Test data')
		self.parser.add_argument('--test_label', type=str, default='testLabels.csv', help='Test label')
		self.parser.add_argument('--normalization', type=bool, default=True, help='Flag for normalization')
		self.parser.add_argument('--pca', type=bool, default=True, help='Flag for impose PCA on data')
		self.parser.add_argument('--num_of_hidden_layer', type=int, default=1, help='Number of hidden layers in the model')
		self.parser.add_argument('--std', type=float, default=0.01, help='Standard deviation for initial weights')
		self.parser.add_argument('--hidden_layers', default=[150], help='Define the number of each hidden layer in a list')
		self.parser.add_argument('--learning_rate', type=float, default=0.0325, help='Learning rate')
		self.parser.add_argument('--learning_rate_decay', type=float, default=0.97, help='Learning rate decay')
		self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value')
		self.parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
		self.parser.add_argument('--reg', type=float, default=1e-4, help='Regularization value')
		self.parser.add_argument('--num_of_iters', type=int, default=1000, help='Number of iteration for training')
		self.parser.add_argument('--model', type=str, default='NN', help='Choose the model for traing')
		self.parser.add_argument('--save_dir', type=str, default='', help='Save directory')

	def parse_params(self):
		self.initial_params()
		args = self.parser.parse_args()

		current_dir = os.getcwd()
		args.dataset_dir = current_dir + '/Dataset/'
		save_dir = current_dir + '/Save_result/'
		args.save_dir = save_dir
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		if args.model == 'NN':
			args.learning_rate = 0.0325
			args.reg = 1e-4
			args.num_of_iters = 1000
			args.std = 0.01
		else:
			args.learning_rate = 0.00325
			args.reg = 1e-3
			args.num_of_iters = 500
			args.std = 0.5

		return args
