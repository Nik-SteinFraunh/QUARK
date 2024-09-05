
import numpy as np

from numpy import genfromtxt

import matplotlib.pyplot as plt


import networkx as nx

""" problemToSolve   = genfromtxt(r'src\modules\applications\optimization\Backtracking\data\graph_coloring.csv', delimiter=';')
num_empty_fields = sum([1 for index in range(len(problemToSolve[1])) if problemToSolve[1] == -1])
print(problemToSolve) """

import pandas as pd

data2 = pd.read_csv(r'src\modules\applications\optimization\Backtracking\data\graph_coloring.csv',delimiter= ";",header = None,  nrows=1)
thisa = data2.values.tolist()[0]
my_list = thisa[0].split(";")
myList = [int(item) for item in my_list]
print(myList)