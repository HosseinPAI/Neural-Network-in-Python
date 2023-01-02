## A Simple Neural Network vs A Gaussian Neural Network in Python
<p align="justify">
This project is a python code without using any prepared library such as Pytorch to implement a neural network, train, and test.
</p>

There are two different models in this project:
* A simple neural network with optional hidden layers with Hing Loss and Relu activation function.
* A Gaussian RBF neural network with optional hidden layers, a specific loss based on the below paper, and GeLu activation function.
<br />https://arxiv.org/abs/1812.03190

<img src="https://openclipart.org/image/2400px/svg_to_png/28580/kablam-Number-Animals-1.png" alt="ReLu Activation Function" width="200"/> <img src="https://openclipart.org/download/71101/two.svg" width="300"/>

ReLu Activation Function             |  GeLu Activation Function
:-----------------------------------:|:---------------------------------------:
![](https://...Dark.png)             |  ![](https://...Ocean.png)


Both networks have been trained on [Fashion MNIST](https://drive.google.com/drive/folders/1_kOFBd-MQY6NJhn5qU1pMLIj1ngTDqTL?usp=sharing) dataset, and you can download it from the below link. After downloading the dataset, you should put CSV files into Dataset folders and then run main.py to see the results. 


<p align="justify">
<br />In params.py, you can find all parameters used in projects and change them to see the effects of various parameters on training and testing models.
<br />In the Save_result folder, you can find figures related to loss and accuracy values during training. In the table below, you see both models' accuracy values on train and test.
</p>

<br />You need Python version 3.* and standard libraries for this project. 


