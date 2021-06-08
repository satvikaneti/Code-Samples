# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:26:03 2020

@author: satvika
"""
import numpy as np
import sys
import csv
import math

def createDict(data):
    d = dict()
    for line in data:
        splitline = line[0].split(" ")
        d[splitline[0]] = splitline[1]
    return d

def sgd(theta, labels, design, epochs):
    alpha = 0.1
    totnum = len(design)
    count = 0    

    while count < epochs:
        for i in range(len(design)):
            dotsum = 0
            #use sparse representation to index into theta without going through each value of theta
            #find dot product first before changing theta
            for k in range(len(design[i])):
                dotsum += theta[design[i][k]]            
            #change theta based on dot product at theta and x_i
            #only changes if x_j (of theta) is in that vector (otherwise the term is 0 and theta doesn't change)
            for k in range(len(design[i])):
                theta[design[i][k]] += (alpha/totnum) * (labels[i] - (math.exp(dotsum)/(1+math.exp(dotsum))))
        count+=1
    return theta

def parseData(data):
    features = []
    labels = []
    for i in range(len(data)):
        #add class to labels
        labels.append(float(data[i][0]))
        #create empty row to make list of lists
        features.append([])
        #only loop through data[i][1:] (ignore the class)
        for k in range(1,len(data[i])):
            #splits apart the "index:1" pair to just get index
            featsplit = data[i][k].split(":")
            features[i].append(int(featsplit[0]))
        #append len(d) as the last index into each vector for the bias term
        features[i].append(len(d))
    #print(features,labels)
    return (np.array(features), np.array(labels))
    
def predict(features, theta):
    predlabels = []
    for i in range(len(features)):
        dotsum = 0
        #finds dot product of theta and x_i
        for k in range(len(features[i])):
            dotsum += theta[features[i][k]]
        #passes into sigmoid function and then uses threshold to make prediction
        predic = math.exp(dotsum)/(1+math.exp(dotsum))
        if predic >= 0.5:
            predlabels.append(1)
        elif predic < 0.5:
            predlabels.append(0)
    #print(predlabels)
    return predlabels 
 
def error(real, pred):
    totnum = len(real)
    numright = 0
    for i in range(len(real)):
       if real[i] == pred[i]:
           numright += 1
    return (totnum - numright)/totnum

def writeLabels(classes, file):
    with open(file, "w") as f:
        for label in classes:
            f.write("%s\n" % (label))
    return

def writeErrors(testerror, trainerror, file):
    with open(file,"w") as f:
        f.write("error(train): %f\nerror(test): %f" % (trainerror, testerror))
    return

if __name__ == '__main__':
    
    #read in elements and load in train and test data
    traininput = sys.argv[1]
    valinput = sys.argv[2]
    testinput = sys.argv[3]
    dictinput = sys.argv[4]
    trainout = sys.argv[5]
    testout = sys.argv[6]
    metricsout = sys.argv[7]
    numepochs = int(sys.argv[8])
    
    traindata = np.array(list(csv.reader(open(traininput, "r"), delimiter = "\t")))
    valdata = np.array(list(csv.reader(open(valinput, "r"), delimiter = "\t")))
    testdata = np.array(list(csv.reader(open(testinput, "r"), delimiter= "\t")))
    dictdata = np.array(list(csv.reader(open(dictinput, "r"), delimiter = "\t")))
    
    #create the dictionary
    d = createDict(dictdata)
    
    #parses data - uses formatted vectors to create representations of sparse vectors
    #creates design matrix with bias term at the end with sparse vectors and a vector of labels
    traindesign, trainreallabels = parseData(traindata)
    testdesign, testreallabels = parseData(testdata)
    
    #initialize theta to be all zeroes and length of dictionary plus bias term    
    theta = np.zeros((len(d)+1))
    
    #perform sgd using sparse representation to learn parameters
    trainparams = sgd(theta, trainreallabels, traindesign, numepochs)
      
    #use learned parameters to predict and output labels with threshold >= 0.5
    trainpredlabels = predict(traindesign, trainparams)
    testpredlabels = predict(testdesign, trainparams)
    
    #write out all files    
    writeLabels(trainpredlabels, trainout)
    writeLabels(testpredlabels, testout)
    writeErrors(error(testreallabels, testpredlabels), error(trainreallabels, trainpredlabels), metricsout)