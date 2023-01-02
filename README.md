## A Simple Neural Network vs A Gaussian Neural Network in Python
<p align="justify">
This project is a python code without using any prepared library such as Pytorch to implement a neural network, train, and test.
</p>

There are two different models in this project:
* A simple neural network with optional hidden layers with Hing Loss and Relu activation function.
* A Gaussian RBF neural network with optional hidden layers, a specific loss based on the below paper, and GeLu activation function.
<br />https://arxiv.org/abs/1812.03190

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/Neural-Network-in-Python/blob/master/.idea/pic/Activation_gelu.png" alt="GeLu Activation Function" width="300" height='250'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/Neural-Network-in-Python/blob/master/.idea/pic/relu.png" alt="GeLu Activation Function" width="300" height='250'/>
  <p align="center">GeLu Activation Function &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ReLu Activation Function </p>


Both networks have been trained on [Fashion MNIST](https://drive.google.com/drive/folders/1_kOFBd-MQY6NJhn5qU1pMLIj1ngTDqTL?usp=sharing) dataset, and you can download it from the below link. After downloading the dataset, you should put CSV files into Dataset folders and then run main.py to see the results. 
<p align="justify">
In params.py, you can find all parameters used in projects and change them to see the effects of various parameters on training and testing models.
</p>
<p align="justify">
In the Save_result folder, you can find figures related to loss and accuracy values during training.
</p>
<p align="justify">
In the table below, you see both models' accuracy values on train and test.
</p>
<br />
<table align="center">
    <thead>
        <tr>
            <th align="center">Model Name </th>
            <th align="center">Max Train Accuracy </th>
            <th align="center">Test Accuracy</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">Simple Neural Netwrok </td>
            <td align="center">95%</td>
            <td align="center">87.2%</td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <td align="center">Gaussian RBF Neural Netwrok</td>
            <td align="center">91%</td>
            <td align="center">85.6%</td>
        </tr>
    </tbody>
</table>

<br />You need Python version 3.* and standard libraries for this project. 
