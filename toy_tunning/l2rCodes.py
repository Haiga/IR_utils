import sys

import numpy as np
import math
from scipy.stats import norm



def getNdcgRelScore(dataset, label):
    web10k = np.array([0, 1, 3, 7, 15])
    letor = np.array([0, 1, 3])
    if dataset == "web10k":
        return web10k[label]
    elif dataset == "letor":
        return letor[label]


def relevanceTest(dataset, value):
    # y % hsPrecisionRel = ("4", 1,  #"3", 1, #"2", 1, # "1", 0, #"0", 0, );
    if dataset == "web10k":
        if value > 1:
            return 1
        else:
            return 0

    elif dataset == "letor":
        # my % hsPrecisionRel = ("2", 1, # "1", 1, # "0", 0 # );
        if value > 0:
            return 1
        else:
            return 0
    elif dataset == "yahoo":
        if value > 0:
            return 1
        else:
            return 0

    return 0


def average_precision(arrayLabel, dataset):
    avgPrecision = 0
    numRelevant = 0
    iPos = 0
    for prec in arrayLabel:
        if relevanceTest(dataset, prec) == 1:
            numRelevant += 1
            avgPrecision += (numRelevant / float(iPos + 1))
        iPos += 1

    if numRelevant == 0:
        return 0.0

    return round(avgPrecision / float(numRelevant), 4)


def dcg(topN, arrayLabel, dataset):
    totalDocs = arrayLabel.shape[0]
    vetDCG = np.array([0.0] * totalDocs, dtype=float)
    # vetDCG = np.array([0.0]*topN, dtype=float)
    vetDCG[0] = getNdcgRelScore(dataset, arrayLabel[0])
    # totalDcos = arrayLabel.shape[0]
    for iPos in range(1, totalDocs):
        # for iPos in range(1, topN):
        r = 0
        if (iPos < totalDocs):
            r = arrayLabel[iPos]
        else:
            r = 0
        if (iPos < 2):
            vetDCG[iPos] = vetDCG[iPos - 1] + getNdcgRelScore(dataset, r)
        else:
            vetDCG[iPos] = vetDCG[iPos - 1] + (getNdcgRelScore(dataset, r) * math.log(2) / math.log(iPos + 1))
    return vetDCG


def ndcg(arrayLabel, dataset):
    # topN = arrayLabel.shape[0]
    topN = 10
    # vetNDCG = np.array([0.0] * arrayLabel.shape[0])

    vetDCG = dcg(topN, arrayLabel, dataset)
    # print vetDCG
    stRates = np.sort(arrayLabel)[::-1]

    bestDCG = dcg(topN, stRates, dataset)

    NDCGAt10 = 0
    if topN > vetDCG.shape[0]:
        return 0.0
    if (bestDCG[topN - 1] != 0):
        NDCGAt10 = vetDCG[topN - 1] / bestDCG[topN - 1]

    return round(NDCGAt10, 4)


def getEvaluation(score, listQ, label, trainFile, metric, resultPrefix):
    dataset = ""
    if "web10k" in trainFile:
        dataset = "web10k"
    elif "dataset" in trainFile:
        dataset = "letor"
    elif "yahoo" in trainFile:
        dataset = "web10k"

    else:
        print("There is no evaluation to this dataset, dataFile: ", trainFile)
        exit(0)

    listQueries = getQueries(listQ)

    lineNum = np.array(range(0, len(label)), dtype=int)
    mat = (np.vstack((np.reshape(score, -1), np.reshape(lineNum, -1)))).T
    apQueries = np.array([0.0] * len(listQueries), dtype=float)
    ndcgQueries = np.array([0.0] * len(listQueries), dtype=float)

    idQ = 0
    MAP = 0
    for query in listQueries:
        matQuery = mat[query]
        matQuery = matQuery[np.argsort(-matQuery[:, 0], kind="mergesort")]
        labelQuery = np.array([0] * matQuery.shape[0], dtype=int)

        i = 0
        for doc in matQuery:
            labelQuery[i] = label[int(doc[1])]
            i += 1

        apQueries[idQ] = average_precision(labelQuery, dataset)
        # print apQueries[idQ]
        ndcgQueries[idQ] = ndcg(labelQuery, dataset)
        # print "NDCG", ndcgQueries[idQ]
        idQ += 1

    if "NDCG" in metric or "ndcg" in metric:
        for predic in ndcgQueries:
            MAP = MAP + predic

    return MAP / idQ, ndcgQueries
    # return ndcgQueries, apQueries



def getQueries(query_id_train):
    queryList = []
    queriesList = []
    # query_id_train.append(-1)
    idQ = -1
    cDoc = 0
    for i in query_id_train:
        if idQ != i:
            if (len(queryList) > 0):
                queriesList.append(queryList[:])
                del queryList[:]

        idQ = i
        queryList.append(cDoc)
        cDoc = cDoc + 1

    queriesList.append(queryList)
    return queriesList



def writeOutFeatureFile(mask, fileName):
    maskList = [int(i) for i in list(mask)]

    with open(fileName, "w") as outs:
        id = 1;
        for d in maskList:
            if d == 1:
                outs.write(str(id) + "\n")
            id = id + 1



def getGeoRisk(mat, alpha):
    ##### IMPORTANT
    # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
    ##############
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk
