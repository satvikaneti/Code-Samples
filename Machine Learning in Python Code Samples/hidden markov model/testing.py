# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:28:13 2020

@author: sneti
"""


import numpy as np
import sys
import csv
import math
import random


emit = [[1./6, 4./6, 1./6],
        [3./8, 1./8, 4./8]]

prior = [4./10, 6./10]

trans = [[1./4, 3./4],
         [3./5, 2./5]]

emit = np.array(emit)
prior = np.array(prior)
trans = np.array(trans)

xdict = {"you":0, "eat":1, "fish":2}
ydict = {"C":0, "D":1}

numy = len(ydict)

row = ["fish", "eat", "you"]
"""

emit = [[1./2, 3./8, 1./8],
        [1./6, 1./6, 2./3]]

prior = [3./5, 2./5]

trans = [[2./5, 3./5],
         [3./4, 1./4]]

emit = np.array(emit)
prior = np.array(prior)
trans = np.array(trans)

xdict = {"win": 0, "league": 1, "Liverpool": 2}
ydict = {"C":0, "D":1}

numy = len(ydict)

row = ["Liverpool", "win", "league"]
"""
allalphas = np.empty((len(row), numy))
allalphastest = np.empty((len(row), numy))

def forward(i, T, allalphas, emit, prior, trans, xdict, ydict, row):
    if i == T:
        return allalphas
    if i == 0:
        B = emit[:,xdict[row[0]]]
        alpha = np.multiply(B, prior)
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)
    else: 
        B = emit[:,xdict[row[i]]]
        A = trans.transpose().dot(allalphas[i-1])
        alpha = np.multiply(B, A)
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)

forward(0, len(row), allalphas, emit, prior, trans, xdict, ydict, row)
print(allalphas)

allbetas = np.empty((len(row), numy))
allbetastest = np.empty((len(row), numy))

def backward(i, T, allbetas, emit, prior, trans, xdict, ydict, row):
    if i == -1:
        return allbetas
    if i == T:
        allbetas[i] = 1
        backward(i-1, T, allbetas, emit, prior, trans, xdict, ydict, row)
    else: 
        B = np.multiply(emit[:,xdict[row[i+1]]], allbetas[i+1])
        #print(B)
        beta = trans.dot(B)
        #print(beta)
        allbetas[i] = beta
        backward(i-1, T, allbetas, emit, prior, trans, xdict, ydict, row)
        
        
backward(len(row)-1, len(row)-1, allbetas, emit, prior, trans, xdict, ydict, row)
print(allbetas)

beta3 = 1

BB2 = np.multiply(emit[:,xdict[row[len(row)-2]]], beta3)
#print(BB2)
beta2 = trans.dot(BB2)

#print(beta2)

BB1 = np.multiply(emit[:,xdict[row[len(row)-3]]], beta2)
#print(BB1)
beta1 = trans.dot(BB1)

#print(beta1)

allbetastest[0] = beta1
allbetastest[1] = beta2
allbetastest[2] = beta3 

#print(allbetastest)





B1 = emit[:,xdict[row[0]]]
alpha1 = np.multiply(B1,prior)

#print(alpha1)

B2 = emit[:,xdict[row[1]]]
A2 = trans.transpose().dot(alpha1)
alpha2 = np.multiply(B2, A2)

#print(alpha2)

B3 = emit[:,xdict[row[2]]]
A3 = trans.transpose().dot(alpha2)
alpha3 = np.multiply(B3,A3)

#print(alpha3)

allalphastest[0] = alpha1
allalphastest[1] = alpha2
allalphastest[2] = alpha3 

#print(allalphastest)
