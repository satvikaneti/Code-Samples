# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 23:10:22 2020

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
    #yhatdict is a dictionary that maps index -> tag for prediction
    xdict = dict()
    ydict = dict()
    yhatdict = dict()
    for i in range(len(index_to_word)):
        xdict[index_to_word[i]] = i
    for i in range(len(index_to_tag)):
        ydict[index_to_tag[i]] = i
        yhatdict[i] = index_to_tag[i]
    return (xdict, ydict, yhatdict)

def logsumtrick(v):
    #log sum trick as outline in the handout
    m = v.max()
    newexps = math.exp(v - m)
    return m + np.log(newexps.sum())

"""
def newforward(i, T, allalphas, emit, prior, trans, xdict, ydict, row):
    if i == T:
        return allalphas
    if i == 0:
        word = row[i]
        B = emit[:,xdict[word]]
        alpha = np.log(B) + np.log(prior)
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)
    else:
        word = row[i]
        B = emit[:,xdict[word]]
        A = trans.transpose().dot(allalphas[i-1])
        alpha = np.log(B) + logsumtrick(np.log(A))
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)
        
def newbackward(i, T, allbetas, emit, prior, trans, xdict, ydict, row):
    if i == -1:
        return allbetas
    if i == T:
        allbetas[i] = np.log(1)
        backward(i-1, allbetas, emit, prior, trans, xdict, ydict, row)
    else:
        word = row[i+1]
        B = 
"""

def forward(i, T, allalphas, emit, prior, trans, xdict, ydict, row):
    #goes forward, creating all the alphas
    
    if i == T:
        #base case
        return allalphas
    if i == 0:
        #use prior to create first alpha, then recurse
        word = row[i]
        B = emit[:,xdict[word]]
        alpha = np.multiply(B, prior)
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)
    else: 
        #create each alpha based on the equation, then recurse
        word = row[i]
        B = emit[:,xdict[word]]
        A = trans.transpose().dot(allalphas[i-1])
        alpha = np.multiply(B, A)
        allalphas[i] = alpha
        forward(i+1, T, allalphas, emit, prior, trans, xdict, ydict, row)
    #end is matrix of all alphas in order 

def backward(i, T, allbetas, emit, prior, trans, xdict, ydict, row):
    #goes backward, creating all the betas
    
    if i == -1:
        #base case
        return allbetas
    if i == T:
        #last beta is always 1, then recurse
        allbetas[i] = 1
        backward(i-1, T, allbetas, emit, prior, trans, xdict, ydict, row)
    else: 
        #create betas using equations, then recurse
        word = row[i+1]
        B = np.multiply(emit[:,xdict[word]], allbetas[i+1])
        beta = trans.dot(B)
        allbetas[i] = beta
        backward(i-1, T, allbetas, emit, prior, trans, xdict, ydict, row)   
    #end result is matrix of all betas in order 
        
def predict(valdata, emit, prior, trans, xdict, ydict, yhatdict):
    
    allpredlabels = []
    alllikelihoods = []
    allpred = []
    allreal = []
    
    for row in valdata:
        #parse rows, pull words for prediction and labels for accuracy
        striprow = [x.split("_")[0] for x in row.split(" ")]
        reallabels = [x.split("_")[1] for x in row.split(" ")]
        allreal.append(reallabels)
        
        #initialize alphas and betas
        allalphas = np.empty((len(striprow), numy))
        allbetas = np.empty((len(striprow), numy))
        
        #forward and backward    
        forward(0, len(striprow), allalphas, emit, prior, trans, xdict, ydict, striprow)
        backward(len(striprow)-1, len(striprow)-1, allbetas, emit, prior, trans, xdict, ydict, striprow)
        
        #calculate loglikelihood per row 
        loglikeli = np.log(allalphas[-1].sum())
        alllikelihoods.append(loglikeli)
        
        #use alphas and betas to create prediction probabilities,
        #then get labels for highest probability
        predictprobs = np.multiply(allalphas, allbetas)
        yhat = np.vectorize(yhatdict.get)(np.argmax(predictprobs, axis = 1))
        allpred.append(list(yhat))
        
        #concatnate predlabels back with words for printing
        predictlabels = [striprow[i]+"_"+yhat[i] for i in range(len(striprow))]
        predictlabels = " ".join(predictlabels)
        allpredlabels.append(predictlabels)
    
    #calculate avg log likelihood and then return 
    avgloglikeli = np.array(alllikelihoods).sum()/len(np.array(alllikelihoods))
        
    return (np.array(allpredlabels), avgloglikeli, allpred, allreal)

def accuracy(real, pred):
    totnum = 0
    numright = 0
    for i in range(len(real)):
        for j in range(len(real[i])):
            if real[i][j] == pred[i][j]:
                numright += 1
            totnum += 1
    return numright/totnum

def writeErrors(avgloglike, accuracy, file):
    with open(file,"w") as f:
        f.write("Average Log-Likelihood: %f\n" % (avgloglike))
        f.write("Accuracy: %f\n" % (accuracy))
    return

if __name__ == '__main__':
    
    #read in elements and load in train and test data
    valinput = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predictfile = sys.argv[7]
    metricfile = sys.argv[8]
    """
    valinput = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_validation.txt"
    index_to_word = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_index_to_word.txt"
    index_to_tag = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_index_to_tag.txt"
    hmmprior = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_hmmprior.txt"
    hmmemit = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_hmmemit.txt"
    hmmtrans = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_hmmtrans.txt"
    predictfile = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_predict_out.txt"
    metricfile = "C:/Users/sneti/Documents/school/ML/hw7/handout/toy_data/toy_metrics_out.txt"
    
    valinput = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_validation.txt"
    index_to_word = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_index_to_word.txt"
    index_to_tag = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_index_to_tag.txt"
    hmmprior = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_hmmprior.txt"
    hmmemit = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_hmmemit.txt"
    hmmtrans = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_hmmtrans.txt"
    predictfile = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_predict_out.txt"
    metricfile = "C:/Users/sneti/Documents/school/ML/hw7/handout/full_data/full_metrics_out.txt"
    """
    
    #open data and code in number of inputs and outputs 
    valdata = open(valinput, "r").read().splitlines()
    index_to_word = open(index_to_word, "r").read().splitlines()
    index_to_tag = open(index_to_tag, "r").read().splitlines()
    hmmprior = np.array(list(csv.reader(open(hmmprior, "r"), delimiter = " ")), dtype = float)
    hmmemit = np.array(list(csv.reader(open(hmmemit, "r"), delimiter = " ")), dtype = float)
    hmmtrans = np.array(list(csv.reader(open(hmmtrans, "r"), delimiter = " ")), dtype = float)
       
    numx = len(index_to_word)
    numy = len(index_to_tag)
    
    #create dictionaries    
    xdict, ydict, yhatdict = createDicts(index_to_word, index_to_tag)
    
    #predict - returns labels, avg logliklihoods, and real labels
    predlabels, avglog, preds, reals = predict(valdata, hmmemit, hmmprior.transpose(), hmmtrans, xdict, ydict, yhatdict)
    
    #write files 
    np.savetxt(predictfile, predlabels, fmt = "%s", newline = "\n")
    writeErrors(avglog, accuracy(reals, preds), metricfile)



    
    