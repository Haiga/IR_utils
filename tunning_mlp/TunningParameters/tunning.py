from __future__ import division
__author__ = 'daniel'

import os
import sys
sys.path.append("/home/daniel/Dropbox/SourceCodes/PythonProjects/")
#pathup = os.path.abspath(os.path.join(
#    os.path.dirname(__file__+"/../"), os.path.pardir))
#sys.path.insert(0, pathup)
#print(sys.path)
import scipy.stats as stats
import pickle
from sklearn import model_selection
import numpy as np
import re
from Utils_L2R.h_l2rMiscellaneous import  getIdFeatureOrder, load_L2R_file, getL2RPrediction, createNewDataset
from Utils_L2R.h_l2rMeasures import getQueries
from collections import defaultdict
from operator import itemgetter
from multiprocessing import Queue, Process
import random

class dataset:
    def __init__(self):
        self.q = None
        self.x = None
        self.y = None

def writingFileWithBestParameter( bestParameter, coll, fold, l2r,  mask, nFeatures):

    listResult = []
    test = dataset()
    train = dataset()
    testFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.test.txt"
    trainFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
    test.x, test.y, test.q = load_L2R_file(testFile, mask)
    train.x, train.y, train.q = load_L2R_file(trainFile, mask)
    listResult.append( getL2RPrediction(l2r, fold, train, test, trainFile, testFile, bestParameter, mask, nFeatures))

    ##################
    #### WRITING THE RESULTS
    finalResult = np.array(listResult)
    finalResult = finalResult.T
    # web10k.MEAN.NDCG@10.result.F5

    print("BEST Executing: usgin parametrs (neur, nlayer, lr) ", bestParameter, ", Fold:", fold, "NDCG: ", np.mean(listResult))

    with open(coll + "." + l2r + ".ndcg.test.Fold" + str(fold), 'w') as f:
        f.write(coll + " fold " + str(fold) + " l2r " + l2r + " parameters " + str(bestParameter) + " " + str(
            parameters[0]) + "\n")
        # print coll, "fold", fold, "l2r", l2r, "parameters", parameters[0]
        for res in finalResult:
            for r in res:
                f.write("{0:.5f}".format(r)),
                # f.write("{0:.5f} ".format(r))
            f.write("mean=>" + str(np.mean(res)) + "\n")


def obtainingDatasetAndPrediction(paralleQueue, coll, fold, l2r, parameter, mask, nFeatures):
    test = dataset()
    train = dataset()


    trainFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.train.txt"
    train.x, train.y, train.q = load_L2R_file(trainFile, mask)
    testFile = "/home/daniel/Colecoes/" + coll + "/Fold" + str(fold) + "/Norm.vali.txt"
    test.x, test.y, test.q = load_L2R_file(testFile, mask)

    result = np.mean(getL2RPrediction(l2r, fold, train, test, trainFile, testFile, parameter, mask, nFeatures))
    print ("Executing: usgin parametrs (neur, nlayer, lr) ", parameter, ", Fold:", fold, "NDCG: ", result)
    paralleQueue.put(result)

if __name__ == "__main__":

    coll = str(sys.argv[1])
    l2r = str(sys.argv[2])
    nFold = int(sys.argv[3])
    nFolds=nFold+1
    nTrees = 300
    nRounds = 800

    print ("Number of Fold ", nFold)

    if "web10k" in coll:
        mask = "1" * 136
        nFeatures = 136
    elif "td_dataset" in coll:
        mask = "1" * 64
        nFeatures = 64
    elif "yahoo" in coll:
        mask = "1"*700
        nFeatures = 700
    elif "web30k" in coll:
        mask = "1" * 136
        nFeatures = 136
    elif "movielens" in coll:
        mask = "1" * 13
        nFeatures = 13
    elif "bibsonomy" in coll:
        mask = "1" * 12
        nFeatures = 12
    elif "youtube" in coll:
        mask = "1" * 13
        nFeatures = 13
    elif "temp" in coll:
        mask = "1" * 64
        nFeatures = 64
    else:
        print ("There is no match to dataset name.", coll)
        sys.exit(0)


    parameters = []
    if l2r == "rf" or l2r =="1":
        parameters.append(300)
    elif nFeatures < 14:
        parameters.append([800, 0.01])
    else:
        #rounds = [100, 300, 400,500,  700, 800, 1000]
        #tolerance = [0.0001, 0.001, 0.01, 0.1]

        # nneurons = 100
        # nlayers = 1
        # lr = 0.001
        # dropout = 0.2
        if l2r == 10 or l2r == "deep":
            nneurons = [ 50, 100, 200 ]
            layers = [1, 2]
            lr = [0.001, 0.01, 0.1]
            for n in nneurons:
                for l in layers:
                    for r in lr:
                        parameters.append([n, l, r])
        else:
            rounds = [3, 5]
            tolerance = [0.1, 0.02]
            #tolerance = [0.01, 0.001, 0.0001 , 0000.1]
            for r in rounds:
                for t in tolerance:
                    parameters.append([r, t])

        paramResults = []
        ##################
        #### TUNNING THE PARAMETERS
        if len(parameters)> 1:
            for param in parameters:
                parallelResult = []
                paralleQueue = Queue()
                jobs = []

                for fold in range(nFold,nFolds):
                    process = Process(target=obtainingDatasetAndPrediction,
                                          args=(paralleQueue, coll, fold, l2r, param, mask, nFeatures))
                    jobs.append(process)
                    process.start()

                for fold in range(nFold, nFolds):
                    parallelResult.append(paralleQueue.get())

                paralleQueue.close()

                for j in jobs:
                    j.join()

                print ("\nParam", param, "Result:", np.mean(parallelResult))
                print ("\n")
                param.append(np.mean(parallelResult))

            parameters = sorted(parameters, key=itemgetter(2), reverse=True)


    print ("The best", parameters[0])
    ##################
    #### USING THE BEST PARAMETER

    #parallelResult = []
    #paralleQueue = Queue()
    jobs = []

    for fold in range(nFold, nFolds):
        process = Process(target=writingFileWithBestParameter,
                          args=(parameters[0], coll, fold, l2r,  mask, nFeatures))
        jobs.append(process)
        process.start()

    #for fold in range(1, nFolds+1):
    #    parallelResult.append(paralleQueue.get())

    paralleQueue.close()

    for j in jobs:
        j.join()

    #print ("mean with Best Parameter =>"+str(np.mean(res)))






