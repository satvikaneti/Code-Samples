# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import sys
import csv
import math
import random

def NNForward(x,y,alpha,beta): 
    
    #find a (linear forward)
    a = np.dot(alpha,x)

    #find z (sigmoid forward)
    z = 1/(1+np.exp(-a))
    z = np.insert(z, 0, 1).transpose()

    #find b (linear forward)
    b = np.dot(beta, z)

    #find yhat (softmax forward)
    yhate = np.exp(b)
    yhat = np.exp(b)/yhate.sum() 

    #find loss (loss forward)
    lossinter = np.multiply(y, np.log(yhat))
    loss = -1*lossinter.sum()     
    
    #return all intermediate values
    return (x,a,b,z,yhat,loss)

def NNBackward(x,y,alpha, beta, o):
    
    #pull out o
    (x,a,b,z,yhat,loss) = o
    
    #dl/db (loss/softmax backward)
    grad_b = yhat - y

    #dl/dbeta (linear backward)
    grad_beta = np.multiply(grad_b, z.transpose())

    #dl/dz (linear backward)
    betastar = np.delete(beta,0,axis = 1)
    grad_z = np.dot(grad_b.transpose(), betastar).transpose()

    #dl/da (sigmoid backward)
    zinter = np.multiply(z, (1-z))
    zinter = np.delete(zinter, 0, axis = 0)
    grad_a = np.multiply(grad_z, zinter)

    #dl/dalpha (linear backward)
    grad_alpha = np.multiply(grad_a, x.transpose())
    
    #return gradients of alpha and beta
    return (grad_alpha, grad_beta)    

def initializeweights(initflag, hiddenunits, numfeats, numoutputs):
    #initialize weights given init flag
    #alpha is hiddenunits x numfeats+1 for the bias and 
    #beta is numoutputs x hiddenunits+1 for the bias
    
    if initflag == 1:
        alpha = np.random.uniform(low = -0.1, high = 0.1, size = (hiddenunits, numfeats+1))
        beta = np.random.uniform(low = -0.1, high = 0.1, size = (numoutputs, hiddenunits+1))
    elif initflag == 2:
        alpha = np.zeros((hiddenunits, numfeats+1))
        beta = np.zeros((numoutputs, hiddenunits+1))
    return (alpha, beta)

def parseData(row):
    #parses each row - creates features and builds a one hot encoding of label
    #transposes to column vector before returns to match written part 
    
    label = row[0]
    y = np.zeros((numoutputs))
    y[label] = 1
    y = np.matrix(y).transpose()
            
    x = row[1:]
    x = np.insert(x, 0, 1)
    x = np.matrix(x).transpose()
    return (x,y)

def crossentropy(data, alpha, beta):
    
    allloss = []
    
    for row in data:
        
        x,y = parseData(row)
        
        #do just forward prop given the weights and append loss
        (x,a,b,z,yhat,loss) = NNForward(x,y,alpha,beta)
        allloss.append(loss)
        
    #return average at the end 
    avg = sum(allloss) / len(data)
    return avg
    
def sgd(traindata, valdata, alpha, beta, numepochs, numoutputs, learningrate):
    
    traincrosses = []
    valcrosses = []
    count = 0
    for epoch in range(numepochs): 
        print(epoch)
        for row in traindata:
            count += 1
            x,y = parseData(row)
            
            #perform forward and back propogation 
            o = NNForward(x,y,alpha,beta)
            grad_alpha, grad_beta = NNBackward(x,y,alpha,beta, o)
            
            #update weights
            alpha = alpha - learningrate*grad_alpha
            beta = beta - learningrate*grad_beta
            
            if count % 10 == 0:
                print(alpha)
                print(beta)
        #find cross entropy using those weights
        traincross = crossentropy(traindata, alpha, beta)
        valcross = crossentropy(valdata, alpha, beta)
        traincrosses.append(traincross)
        valcrosses.append(valcross)
    
    #after all epochs, return final weights and list of cross entropy losses 
    return (alpha, beta, traincrosses, valcrosses)

def predict(data, alpha, beta):
    
    predlabels = []
    
    for row in data:
        
        x,y = parseData(row)
        
        #find values by doing NNForward and then find the maximum of the array
        (x,a,b,z,yhat,loss) = NNForward(x,y,alpha,beta)
                        
        #return index of max value (which is the label)
        ind = np.where(yhat == yhat.max())
        predlabels.append(ind[0])
        
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
            f.write("%s\n" % (label[0]))
    return

def writeErrors(traincross, valcross, trainerror, valerror, file):
    with open(file,"w") as f:
        for epoch in range(len(traincross)):
            f.write("epoch=%d crossentropy(train): %f\n" % (epoch+1, traincross[epoch]))
            f.write("epoch=%d crossentropy(validation): %f\n" % (epoch+1, valcross[epoch]))
        f.write("error(train): %f\nerror(validation): %f" % (trainerror, valerror))
    return

if __name__ == '__main__':
    """
    #read in elements and load in train and test data
    traininput = sys.argv[1]
    valinput = sys.argv[2]
    trainout = sys.argv[3]
    valout = sys.argv[4]
    metricsout = sys.argv[5]
    numepochs = int(sys.argv[6])
    hiddenunits = int(sys.argv[7])
    initflag = int(sys.argv[8])
    learningrate = float(sys.argv[9])
    
    """
    traininput = "C:/Users/Allison Michalowski/Downloads/hw5/handout/smallTrain.csv"
    valinput = "C:/Users/Allison Michalowski/Downloads/hw5/handout/smallValidation.csv"
    trainout = "C:/Users/Allison Michalowski/Downloads/hw5/handout/smalltrainlabels.txt"
    valout = "C:/Users/Allison Michalowski/Downloads/hw5/handout/smallvallabels.txt"
    metricsout = "C:/Users/Allison Michalowski/Downloads/hw5/handout/smallmetrics.txt"
    numepochs = 1
    hiddenunits = 4
    initflag = 2
    learningrate = .1
    
    
    traindata = np.array(list(csv.reader(open(traininput, "r"), delimiter = ",")), dtype = int)
    valdata = np.array(list(csv.reader(open(valinput, "r"), delimiter = ",")), dtype = int)

    #hard doing number of outputs and defining number of features
    numoutputs = 10
    numfeats = len(traindata[0]) - 1
    
    #pulling out real labels
    #trainreallabels = traindata[:,0]
    #valreallabels = valdata[:,0]
    
    #init weights given initflag   
    alphastart, betastart = initializeweights(initflag, hiddenunits, numfeats, numoutputs)
    
    #perform SGD to find weights        
    alpha, beta, traincross, valcross = sgd(traindata, valdata, 
                                            alphastart, betastart, 
                                            numepochs, numoutputs, learningrate)
    
    #predict new labels given trained weights
    #trainpredlabels = predict(traindata, alpha, beta)
    #valpredlabels = predict(valdata, alpha, beta)
        
    #write files
    #writeLabels(trainpredlabels, trainout)
    #writeLabels(valpredlabels, valout)
    
    #writeErrors(traincross, valcross, 
                #error(trainreallabels,trainpredlabels), 
                #error(valreallabels,valpredlabels), 
                #metricsout)
    
    




    