# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:57:21 2020

@author: satvika
"""
import numpy as np
import sys
import csv

class TreeNode:
    def __init__(self, attrindex, val, vals, counts, vote, data):
        self.attrindex = attrindex
        self.val = val
        self.vals = vals
        self.counts= counts
        self.vote = vote
        self.data = data
        self.leftNode = None
        self.rightNode = None
        #need labels and outcomes at each node too? 
    
    #print for debugging purposes
    def printTreeNode(self, header):
        if self.attrindex == None:
            print("Root Node")
        else:
            print(header[self.attrindex] + " at index " + str(self.attrindex) + " = " + self.val + ". Majority vote is " + self.vote + ". Counts are " + str(self.counts[0]) + " for " + self.vals[0] + " and " + str(self.counts[1]) + " for " + self.vals[1])
            if self.leftNode != None:
                print("|" + header[self.leftNode.attrindex] + "at index" + str(self.leftNode.attrindex) + " = " + self.leftNode.val + ". Majority vote is " + self.leftNode.vote + ". Counts are " + str(self.leftNode.counts[0]) + " for " + self.leftNode.vals[0] + " and " + str(self.leftNode.counts[1]) + " for " + self.leftNode.vals[1])
            if self.rightNode != None:
                print("|" + header[self.rightNode.attrindex] + "at index" + str(self.rightNode.attrindex) + " = " + self.rightNode.val + ". Majority vote is " + self.rightNode.vote + ". Counts are " + str(self.rightNode.counts[0]) + " for " + self.rightNode.vals[0] + " and " + str(self.rightNode.counts[1]) + " for " + self.rightNode.vals[1])
        
    def createStump(parent, left, right):
        parent.leftNode = left
        parent.rightNode = right
        return parent 

def entropy(data):
    #for each class, find probability of being in that class * log2(prob)
    #then add together
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

def specCondEntropy(data, attrindex, attrval):
    #for a given attribute value, find probability of each class within that attribute
    classindex = len(data[0])-1
    
    #filter data to just where we have attrval, then find entropy within that
    newdata = data[data[:,attrindex]==attrval]
    
    (vals, counts) = np.unique(newdata[:,classindex], return_counts=True)
    totnum = len(newdata)

    entropy = 0
    
    for i in range(len(vals)):
        prob = counts[i]/totnum
        if prob == 0:
            entropy += 0
        else:
            entropy += prob*np.log2(prob)
    
    return entropy * -1

def condEntropy(data, attrindex):
    #for each class, find probability based on each attribute in the attribute values    
    totnum = len(data)
    
    (vals, counts) = np.unique(data[:,attrindex], return_counts=True)
    
    entropy = 0
    
    #for each attribute, pass into specCondEntropy()
    for i in range(len(vals)):
        prob = counts[i]/totnum
        entropy += prob*specCondEntropy(data,attrindex,vals[i])
   
    return entropy
    
def mutualInfo(index, data):
    return entropy(data) - condEntropy(data, index)

def bestAttribute(data):
    #finds the mutual info for each attribute
    
    mutinfolist = np.empty((1,len(data[0])-1))
    for i in range(len(data[0])-1):
        mutinfolist[0][i] = mutualInfo(i, data)
    
    #find biggest one and return if > 0
    index = np.argmax(mutinfolist)
    
    if mutinfolist[0][index] > 0:
        return index
    else:
        return None 

def getMajority(data):
    #finds and returns the values, counts of each value, and the majority vote of the classes of a dataset
    classindex = len(data[0])-1
    
    (vals, counts) = np.unique(data[:,classindex], return_counts=True)
    if len(vals) == 1:
        vote = vals[0]
    else:
        vote = vals[1] if counts[1]>=counts[0] else vals[0]
    return (vals, counts, vote)

def createStump(root, splitindex):
        
    #find left and right datasets, np.unique returns ordered so always in the same direction
    (left, right) = np.unique(root.data[:,splitindex])
    (leftData, rightData) = (np.empty((0,len(root.data[0]))), np.empty((0,len(root.data[0]))))
    leftData = np.vstack((leftData, root.data[root.data[:,splitindex] == left]))
    rightData = np.vstack((rightData, root.data[root.data[:,splitindex] == right]))

    #create nodes with root node taking all the same attributes it had before
    #getMajority returns (vals, counts, vote) in that order to preserve class instances 
    rootNode = TreeNode(root.attrindex, root.val, root.vals, root.counts, root.vote, root.data)
    leftNode = TreeNode(splitindex, left, getMajority(leftData)[0], getMajority(leftData)[1], getMajority(leftData)[2], leftData)
    rightNode = TreeNode(splitindex, right, getMajority(rightData)[0], getMajority(rightData)[1], getMajority(rightData)[2], rightData)
    
    #then connect them to create a decision stump and return
    stump = TreeNode.createStump(rootNode, leftNode, rightNode)
    return stump

def buildTree(stump, maxDepth, currentDepth):
    #find best attribute
    splitindex = bestAttribute(stump.data)
    
    if (maxDepth <= currentDepth) or (splitindex == None):
    #base case, just return the stump we're at without adding left or right
        return stump
    else:
    #create a stump using the createStump() function from HW1 and then recurse on left and right nodes
        stump = createStump(stump, splitindex)
        stump.leftNode = buildTree(stump.leftNode, maxDepth, currentDepth + 1)
        stump.rightNode = buildTree(stump.rightNode, maxDepth, currentDepth + 1)
        return stump

def testTree(stump, row, predclasses):
    if (stump.leftNode == None) and (stump.rightNode == None):
    #base case - append the vote here
        return predclasses.append(stump.vote)
    else:
        if row[stump.leftNode.attrindex] == stump.leftNode.val:
            testTree(stump.leftNode, row, predclasses)
        elif row[stump.rightNode.attrindex] == stump.rightNode.val:
            testTree(stump.rightNode, row, predclasses)
    
def testTreeWrapper(stump, data, predclasses):
    #wrapper function for testTree()
    #for each row in the data, traverses the tree to find the right class
    for row in data:
        testTree(stump, row, predclasses)
    return predclasses
       
def error(data):
    predclass = len(data[0])-1
    realclass = len(data[0])-2
    totnum = len(data)
    numright = 0
    for row in data:
       if row[predclass] == row[realclass]:
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

def printTree(stump, header, d):
    #prints the attribute and value at each node, with printing counts at each node as well
    if (stump.leftNode == None) and (stump.rightNode == None):
    #leaf node, base case - if statement for how many values there are in the node
        if len(stump.vals) == 1:
            print("|"*d + str(header[stump.attrindex]) + " = " + str(stump.val) + ": ["
              + str(stump.counts[0]) + " " + str(stump.vals[0]) + "]")
        else:
            print("|"*d + str(header[stump.attrindex]) + " = " + str(stump.val) + ": ["
                  + str(stump.counts[0]) + " " + str(stump.vals[0]) + "/" + str(stump.counts[1]) + " " + str(stump.vals[1]) + "]")
    else:
    #print node out, then recurse
        if d != 1:
            if len(stump.vals) == 1:
                print("|"*d + str(header[stump.attrindex]) + " = " + str(stump.val) + ": ["
                      + str(stump.counts[0]) + " " + str(stump.vals[0]) + "]")
            else:
                print("|"*d + str(header[stump.attrindex]) + " = " + str(stump.val) + ": ["
                      + str(stump.counts[0]) + " " + str(stump.vals[0]) + "/" + str(stump.counts[1]) + " " + str(stump.vals[1]) + "]")
        printTree(stump.leftNode, header, d+1)
        printTree(stump.rightNode, header, d+1)
        
def printTreeWrapper(stump, header):
    #wrapper function for recursive printTree() function to print out the initial counts
    if len(stump.vals) == 1:
        print("[" + str(stump.counts[0]) + " " + str(stump.vals[0]) + "]")
    else:
        print("[" + str(stump.counts[0]) + " " + str(stump.vals[0]) + "/" + str(stump.counts[1]) + str(stump.vals[1]) + "]")
    
    printTree(tree, header, 1)

if __name__ == '__main__':
    
    #read in elements and load in train and test data
    traininput = sys.argv[1]
    testinput = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainout = sys.argv[4]
    testout = sys.argv[5]
    metricsout = sys.argv[6]
    
    traindata = np.array(list(csv.reader(open(traininput, "r"), delimiter = "\t")))
    testdata = np.array(list(csv.reader(open(testinput, "r"), delimiter= "\t")))

    #store and then delete headers
    header = traindata[0]
    
    traindata = np.delete(traindata, (0), axis = 0)
    testdata = np.delete(testdata, (0), axis = 0)
    
    #build first root node     
    #getMajority returns (vals, counts, vote) in that order, which matches the class instance set up 
    momRoot = TreeNode(None, None, getMajority(traindata)[0], getMajority(traindata)[1], getMajority(traindata)[2], traindata)
    
    #send that node into buildTree to build the tree
    tree = buildTree(momRoot, maxDepth, 0)
    
    #testing test and train data each on new tree
    predclassestest = testTreeWrapper(tree, testdata, [])
    newtestdata = np.hstack((testdata, np.array(predclassestest)[:, None]))
    
    predclassestrain = testTreeWrapper(tree, traindata, [])
    newtraindata = np.hstack((traindata, np.array(predclassestrain)[:, None]))

    #writing out files using those test and train datasets
    writeLabels(predclassestrain, trainout)
    writeLabels(predclassestest, testout)
    
    writeErrors(error(newtestdata), error(newtraindata), metricsout)
    
    #print tree
    printTreeWrapper(tree, header)
