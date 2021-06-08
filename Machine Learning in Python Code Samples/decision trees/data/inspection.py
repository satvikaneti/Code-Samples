# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 01:44:31 2020

@author: satvika
"""


import numpy as np
import sys
import csv

#define classes

def entropy(data):
    classindex = len(data[0])-1
    totnum = len(data)
    
    (vals, counts) = np.unique(data[:,classindex], return_counts=True)
    entropy = 0
    
    for i in range(len(vals)):
        prob = counts[i]/totnum
        if prob == 0:
            entropy += 0
        else:
            entropy += prob*np.log2(prob)
    return entropy * -1

def getMajority(data):
    classindex = len(data[0])-1
    
    (vals, counts) = np.unique(data[:,classindex], return_counts=True)
    if len(vals) == 1:
        vote = vals[0]
    else:
        vote = vals[1] if counts[1]>=counts[0] else vals[0]
    return (vote, vals, counts)

def writeFile(entropy, error, file):
    with open(file,"w") as f:
        f.write("entropy: %f\nerror: %f" % (entropy, error))
    return
    
if __name__ == '__main__':
    #read in elements and load in train and test data
    
    traininput = sys.argv[1]
    inspectout = sys.argv[2]

    traindata = np.array(list(csv.reader(open(traininput, "r"), delimiter = "\t")))
    traindata = np.delete(traindata, (0), axis = 0)
    
    entropyval = entropy(traindata)
    (majvote, vals, counts) = getMajority(traindata)
    error = counts[0]/len(traindata) if majvote==vals[1] else counts[1]/len(traindata)
    
    writeFile(entropyval, error, inspectout)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    