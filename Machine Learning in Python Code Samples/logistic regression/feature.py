# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:53:14 2020

@author: satvika
"""
import numpy as np
import sys
import csv

def createDict(data):
    #creates dictionary
    d = dict()
    for line in data:
        splitline = line[0].split(" ")
        d[splitline[0]] = splitline[1]
    return d

def modelOne(data, d):
    n = []
    #splits the data from the class and creates a list of lists
    #where first one is the class and the rest are the words we haven't seen
    for i in range(len(data)):
        splitline = data[i][1].split(" ")
        seenfeats = set()
        n.append([data[i][0]])
        for feat in splitline:
            if (feat in d) and (feat not in seenfeats):
                n[i].append(d[feat] + ":" + str(1))
                seenfeats.add(feat)                
    return np.array(n)
                
def modelTwo(data, d, threshold):
    n = []
    #same thing as above but with the added constraint of making sure the count is less than four before adding
    for i in range(len(data)):
        splitline = data[i][1].split(" ")
        seenfeats = set()
        n.append([data[i][0]])
        for feat in splitline:
            if (feat in d) and (feat not in seenfeats):
                featcount = splitline.count(feat)
                if (featcount < 4):
                    n[i].append(d[feat] + ":" + str(1))
                seenfeats.add(feat) 
    return np.array(n) 

def writeFormatted(n, file):
    with open(file, "w") as f:
        for row in n:
            for col in row:
                f.write("%s\t" % (col))
            f.write("\n")
    return    

if __name__ == '__main__':
    
    #read in elements and load in train and test data
    traininput = sys.argv[1]
    valinput = sys.argv[2]
    testinput = sys.argv[3]
    dictinput = sys.argv[4]
    trainout = sys.argv[5]
    valout = sys.argv[6]
    testout = sys.argv[7]
    featflag = int(sys.argv[8])
    
    traindata = np.array(list(csv.reader(open(traininput, "r"), delimiter = "\t")))
    valdata = np.array(list(csv.reader(open(valinput, "r"), delimiter = "\t")))
    testdata = np.array(list(csv.reader(open(testinput, "r"), delimiter= "\t")))
    dictdata = np.array(list(csv.reader(open(dictinput, "r"), delimiter = "\t")))
    
    #create dictionary
    d = createDict(dictdata)
    
    if featflag == 1:
        trainbagged = modelOne(traindata, d)
        writeFormatted(trainbagged, trainout)
        
        valbagged = modelOne(valdata, d)
        writeFormatted(valbagged, valout)
        
        testbagged = modelOne(testdata, d)
        writeFormatted(testbagged, testout)
        
    elif featflag == 2:
        traintrimmed = modelTwo(traindata, d, 4)
        writeFormatted(traintrimmed, trainout)
        
        valtrimmed = modelTwo(valdata, d, 4)
        writeFormatted(valtrimmed, valout)
        
        testtrimmed = modelTwo(testdata, d, 4)
        writeFormatted(testtrimmed, testout)
        
    