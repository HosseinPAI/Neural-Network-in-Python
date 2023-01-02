# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:07:20 2020

@author: MrHossein
"""
from train_test import train_test
from params import parameters
import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":
    # set parameters
    param_class = parameters()
    params = param_class.parse_params()

    # Train and Test Models
    train_test(params)






