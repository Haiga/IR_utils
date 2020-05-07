from __future__ import division
__author__ = 'daniel'

import sys, os
import os.path

#pathup = os.path.abspath(os.path.join(
#    os.path.dirname(__file__+"/../../"), os.path.pardir))
#sys.path.insert(0, pathup)

#print(sys.path)
import time
import scipy.stats as stats
import pickle
import numpy as np
import re
from h_l2rMiscellaneous import getL2RPrediction
from h_l2rMiscellaneous import load_L2R_file
import h_l2rMeasures as measures

from collections import defaultdict


class basicStructure:
    def __init__(self):
        self.marginal = None
        self.mat = None
        self.pvalue = None


class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None

def executeL2R(coll, foldLimit, l2r, inputFile, nexec, printResult):

    nTrees = 300

    for fold in range(1, foldLimit + 1):

        test = dataset()
        train = dataset()
        testFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.test.txt"
        trainFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"

        with open(inputFile + "F" + str(fold) + ".ind", 'r') as f:
            for line in f:
                mask = line
        mask = mask.rstrip()
        print "Mask:" + "[" + mask + "]"

        nFeatures = len(mask)

        test.x, test.y, test.q = load_L2R_file(testFile, mask)
        train.x, train.y, train.q = load_L2R_file(trainFile, mask)

        ndcg = np.array([0.0] * len(measures.getQueries(test.q)))

        for exe in range(nexec):
            print "nFeatyures", nFeatures
            ndcg = ndcg + getL2RPrediction(l2r, fold + exe, train, test, trainFile, testFile, nTrees, mask, nFeatures)

        if printResult != 1:
            with open(inputFile + "F" + str(fold) + ".rf.test.prediction_or", 'w') as f:
                for i in ndcg:
                    f.write(str(i) + "\n")

    ndcg = ndcg / nexec
    return ndcg
if __name__ == "__main__":


    coll = str(sys.argv[1])
    foldLimit = int(sys.argv[2])
    l2r = str(sys.argv[3])
    inputFile = str(sys.argv[4])
    printResult = int(sys.argv[5])
    nExecs = int(sys.argv[6])

    print ("Processing FS " + coll + "Fold"+ str(foldLimit))

    ndcg= executeL2R(coll, foldLimit, l2r, inputFile, nExecs, printResult)
    if printResult == 1:
        print "Precision", np.mean(ndcg)
