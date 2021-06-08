# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:04:48 2020

@author: sneti
"""

import numpy as np
import sys
import csv
import math
import random

def createDicts(index_to_word, index_to_tag):
    #xdict is a dictionary that maps word -> index
    #ydict is a dictionary that maps tag -> index 
    xdict = dict()
    ydict = dict()
    for i in range(len(index_to_word)):
        xdict[index_to_word[i]] = i
    for i in range(len(index_to_tag)):
        ydict[index_to_tag[i]] = i
    return (xdict, ydict)

def parseData(traindata, numx, numy, xdict, ydict):    
    
    #initialize matrixes 
    Bcounts = np.zeros((numy,numx))
    ycounts = np.zeros((numy,1))
    Acounts = np.zeros((numy, numy))
    
    for i in range(len(traindata)):
        #parse data, use first word to create priors 
        row = traindata[i].split(" ")
        word, tag = row[0].split("_")
        ycounts[ydict[tag]] += 1
        for j in range(len(row)):
            #parse data
            #increment for the i,j pair that corresponds to the word and tag
            word, tag = row[j].split("_")
            Bcounts[ydict[tag]][xdict[word]] += 1
            #calculate trans matrix based on current tag and the next tag
            if j != len(row)-1:
                word2, tag2 = row[j+1].split("_")
                Acounts[ydict[tag]][ydict[tag2]] += 1
                
    #increment all by 1 and divide by total across rows to get probability 
    Bcounts = Bcounts + 1
    emit = Bcounts/Bcounts.sum(axis = 1)[:,None]
    
    ycounts = ycounts + 1 
    prior = ycounts/np.sum(ycounts)
    
    Acounts = Acounts + 1
    trans = Acounts/Acounts.sum(axis = 1)[:,None]
    
    #return 
    return (prior, trans, emit)    

if __name__ == '__main__':
    
    #read in elements and load in train and test data
    traininput = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmpriorout = sys.argv[4]
    hmmemitout = sys.argv[5]
    hmmtransout = sys.argv[6]
    
    """
    traininput = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_train.txt"
    index_to_word = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_index_to_word.txt"
    index_to_tag = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_index_to_tag.txt"
    hmmpriorout = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_prior_out.txt"
    hmmemitout = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_emit_out.txt"
    hmmtransout = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_trans_out.txt"
    
    traininput = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_train.txt"
    index_to_word = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_index_to_word.txt"
    index_to_tag = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_index_to_tag.txt"
    hmmpriorout = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_prior_out.txt"
    hmmemitout = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_emit_out.txt"
    hmmtransout = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_trans_out.txt"
    """
    
    #open datasets and load in num inputs and outputs for matrices 
    traindata = open(traininput, "r").read().splitlines()
    index_to_word = open(index_to_word, "r").read().splitlines()
    index_to_tag = open(index_to_tag, "r").read().splitlines()
    
    numx = len(index_to_word)
    numy = len(index_to_tag)
    
    #create dictionaries for easy access to indexes 
    xdict, ydict = createDicts(index_to_word, index_to_tag)
    
    #parse data into the prior, trans, emit matrixes
    prior, trans, emit = parseData(traindata, numx, numy, xdict, ydict)
    
    #write into files
    np.savetxt(hmmpriorout, prior, delimiter = " ", newline = "\n")
    np.savetxt(hmmemitout, emit, delimiter = " ", newline = "\n")
    np.savetxt(hmmtransout, trans, delimiter = " ", newline = "\n")
    
    