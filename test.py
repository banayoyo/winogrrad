#https://www.cnblogs.com/sunshine-blog/p/8477523.html
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint
from operator import mul
#https://blog.csdn.net/DeniuHe/article/details/77758710
from functools import reduce


import tensorflow as tf
import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import pandas as pd


def test_win():
#    define the data shape
    n_size = 1
    h_size = 105
    w_size = h_size
    c_size = 128
    k_size = c_size

#    define the data matrix
    data_ori = np.random.random(size=(n_size, c_size, h_size, w_size))
    kernal = np.random.random(size=(k_size, c_size, h_size, w_size))
    result = np.zeros(shape=[n_size, k_size, h_size, w_size])
    

    if fractionsIn == FractionsInF:
        print ("fractions = ")
        pprint(f)
        print ("")

def main():
    test_win()
    print ("======================= ")

#debugfile('C:/Users/admin/Desktop/Git_Repo/TF/winogrrad/wincnn.py')
if __name__ == '__main__':
    main()
