## A Simple Neural Network vs A Gaussian Neural Network in Python
  This project is a python code without using any prepared library such as Pytorch to implement a neural network, train, and test.
  There are two different models in this project:
    - A simple neural network with optional hidden layers with Hing Loss and Relu activation function.
    - A Gaussian RBF neural network with a specific loss based on the below paper and GeLu activation function.
      https://arxiv.org/abs/1812.03190
   
  Both networks have been trained on [Fashion MNIST]() dataset, and you can download it from the below link. After downloading the dataset, you should put CSV files into         Dataset folders and then run main.py to see the results. 
  Fashion MNIST Dataset: 
  In params.py, you can find all parameters used in projects and change them to see the effects of various parameters on training and testing models.
  In the Save_result folder, you can find figures related to loss and accuracy values during training. In the table below, you see both models' accuracy values on train and     test.

  You need Python version 3.* and standard libraries for this project. 

