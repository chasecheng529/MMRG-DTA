import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import os
from tqdm import tqdm
from math import exp
os.environ['PYTHONHASHSEED'] = '0'

import time
from copy import deepcopy
from Model import MMRGNet
from sklearn.metrics import roc_auc_score, accuracy_score

from Log import LogModule
from Metrics import *
from DataHelper import *
from Arguments import argparser
from collections import Counter

PARA = argparser()
device = torch.device("cuda:" + str(PARA.GPU))

def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))
        indices = indices.difference(fold)
        folds.append(fold)

    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize

    return folds

def get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows:
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs
    drugfolds = get_random_folds(drugcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind]
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist()
        folds.append(fold)
    return folds

def get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds

def GenLoss(recon_x, x, mu, logvar):
    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(cr_loss + KLD)

def InitWeights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

def train(trainDataLoader, model, PARA, optimizer, lamda):
    model.train()
    lossFunc = nn.MSELoss()
    with tqdm(trainDataLoader) as t:
        MSELossList = []
        drugLossList = []
        targetLossList = []
        trainLossList = []

        for drugSMILES, drugAdj, drugDegree, proteinSeq, affinity in t:
            drugSMILES = drugSMILES.cuda(device)
            proteinSeq = proteinSeq.cuda(device)
            drugDegree = drugDegree.cuda(device)
            affinity = affinity.cuda(device)
            optimizer.zero_grad()
            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, drugAdj, drugDegree, proteinSeq, PARA)
            affinityLoss = lossFunc(pre_affinity, affinity)
            drugLoss = GenLoss(new_drug, drug, mu_drug, logvar_drug)
            proteinLoss = GenLoss(new_target, target, mu_target, logvar_target)

            loss = affinityLoss + 10 ** lamda * (drugLoss + PARA.maxDrugLen / PARA.maxProteinLen * proteinLoss)
            loss.backward()
            optimizer.step()

            MSELossList.append(affinityLoss.item())
            drugLossList.append(drugLoss.item())
            targetLossList.append(proteinLoss.item())
            trainLossList.append(loss.item())

            t.set_postfix(TrainLoss=np.mean(trainLossList), MSE=np.mean(MSELossList))
        logger.LogInfoWithArgs("Train:",MSE = np.mean(MSELossList), drugLoss=np.mean(drugLossList), targetLoss= np.mean(targetLossList), totalLoss=np.mean(trainLossList))
    return model

def test(testDataLoader, model, PARA):
    model.eval()
    lossFunc = nn.MSELoss()
    MAELossFunc = nn.L1Loss()
    affinities = []
    pre_affinities = []
    with torch.no_grad():
        for i,(drugSMILES, drugAdj, drugDegree, proteinSeq, affinity) in enumerate(testDataLoader):
            drugSMILES = drugSMILES.cuda(device)
            proteinSeq = proteinSeq.cuda(device)
            drugDegree = drugDegree.cuda(device)

            pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, drugAdj, drugDegree, proteinSeq, PARA)
            
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()

        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        MSELoss = lossFunc(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        MAELoss = MAELossFunc(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        cIndex = GetCIndex(affinities,pre_affinities)
        rm2 = GetRM2(affinities, pre_affinities)
    return cIndex, MSELoss, MAELoss, rm2


def ParameterSearch(PARA, valDataLoader, trainDataLoader):
    proteinFilterLen = PARA.proteinFilterLen
    drugFilterLen = PARA.drugFilterLen
    lamda = PARA.lamda

    eachSearchRefCindex = []
    for eachProteinFilterLen in proteinFilterLen:
        for eachDrugFilterLen in drugFilterLen:
            for eachLamda in lamda:
                logger.LogInfoWithStr("INFO","============================SEARCH=========================")
                model = MMRGNet(PARA, eachProteinFilterLen, eachDrugFilterLen).cuda(device)
                model.apply(InitWeights)
                optimizer = optim.Adam(model.parameters(), lr = 0.001)
                searchRefCindex = 0
                for epochind in range(PARA.numEpoch):
                    model = train(trainDataLoader, model, PARA, optimizer, eachLamda)
                    cIndex, _, _, _ = test(valDataLoader, model, PARA)
                    searchRefCindex = max(searchRefCindex, cIndex)
                eachSearchRefCindex.append(searchRefCindex)
                logger.LogInfoWithStr("INFO","CI :{} --> PARA: ProteinFilterLen: {}, DrugFilterLen: {}, Lamda: {}".format(searchRefCindex, eachProteinFilterLen, eachDrugFilterLen, eachLamda))
    return eachSearchRefCindex

# This function is used to run 6Fold Validation, each splited folds will be used 6 times!
def RunNFoldExperiment(XD, XAdj, XDegree, XT, Y, label_row_inds, label_col_inds, PARA, labeled_sets, val_sets, test_sets):
    foldParaSearchResult = []
    bestProteinFilterLen, bestDrugFilterLen, bestLamda = 0, 0, 0
     # If only one choice in parameter list, do not run parameter search
    if len(PARA.proteinFilterLen) == 1 and len(PARA.drugFilterLen) == 1 and len(PARA.lamda) == 1:
        logger.LogInfoWithStr('INFO', 'No need to run parameter search')
        bestProteinFilterLen, bestDrugFilterLen, bestLamda = PARA.proteinFilterLen[0], PARA.drugFilterLen[0], PARA.lamda[0]
    else:
        for foldind in range(len(val_sets)):
            logger.LogInfoWithStr("FOLD:{}".format(foldind),"============================PARAMETER SEARCH=========================")
            valinds = val_sets[foldind]
            labeledinds = labeled_sets[foldind]
            testinds = test_sets[foldind]

            terows = label_row_inds[testinds]
            tecols = label_col_inds[testinds]
            test_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, terows, tecols)

            trrows = label_row_inds[labeledinds]
            trcols = label_col_inds[labeledinds]
            train_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, trrows, trcols)

            valrows = label_row_inds[valinds]
            valcols = label_col_inds[valinds]
            val_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, valrows, valcols)


            train_loader = DataLoader(dataset = train_dataset, batch_size = PARA.batchSize, shuffle = True)
            test_loader = DataLoader(dataset = test_dataset, batch_size = PARA.batchSize)
            val_loader = DataLoader(dataset = val_dataset, batch_size = PARA.batchSize)

            foldParaSearchResult.append(ParameterSearch(PARA, val_loader, train_loader))
        
        # Get Parameter search result
        foldParaSearchResultArray = np.array(foldParaSearchResult)
        foldParaSearchSum = list(foldParaSearchResultArray.sum(axis = 0))
        bestPointer = foldParaSearchSum.index(max(foldParaSearchSum))
        pointer = 0
        for eachProteinFilterLen in PARA.proteinFilterLen:
            for eachDrugFilterLen in PARA.drugFilterLen:
                for eachLamda in PARA.lamda:
                    if bestPointer == pointer:
                        bestProteinFilterLen, bestDrugFilterLen, bestLamda = eachProteinFilterLen, eachDrugFilterLen, eachLamda
                        pointer += 1
                    else:
                        pointer += 1
                        continue
    logger.LogInfoWithStr("INFO","BEST PARA: ProteinFilterLen: {}, DrugFilterLen: {}, Lamda: {}".format(bestProteinFilterLen, bestDrugFilterLen, bestLamda))

    logger.LogInfoWithStr("INFO","============================TRAINING=========================")
    crossValMSE = []
    crossValMAE = []
    crossValCI = []
    crossValRM2 = []
    for foldind in range(len(val_sets)):
        logger.LogInfoWithStr("FOLD:{}".format(foldind),"==========================================================")
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]
        testinds = test_sets[foldind]

        terows = label_row_inds[testinds]
        tecols = label_col_inds[testinds]
        test_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, terows, tecols)

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]
        train_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, trrows, trcols)

        valrows = label_row_inds[valinds]
        valcols = label_col_inds[valinds]
        val_dataset = prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, valrows, valcols)


        train_loader = DataLoader(dataset = train_dataset, batch_size = PARA.batchSize, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = PARA.batchSize)
        val_loader = DataLoader(dataset = val_dataset, batch_size = PARA.batchSize)

        model = MMRGNet(PARA, bestProteinFilterLen, bestDrugFilterLen).cuda(device)
        model.apply(InitWeights)
        optimizer = optim.Adam(model.parameters(), lr = 0.001)

        refMSE = np.inf
        for epochind in range(PARA.numEpoch):
            model = train(train_loader, model, PARA, optimizer, bestLamda)
            valCindex, valMSE, valMAE, valR2 = test(test_loader, model, PARA)
            logger.LogInfoWithStr('VAL', 'Epoch:{}, MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(epochind, valMSE, valMAE, valCindex, valR2))
            if refMSE >= valMSE:
                refMSE = valMSE
                torch.save(model, 'checkpoint-{}.pth'.format(PARA.gitNode))

        model = torch.load('checkpoint-{}.pth'.format(PARA.gitNode))
        cIndex, mse, mae, rm2 = test(test_loader, model, PARA)
        logger.LogInfoWithStr('FOLD Result', 'MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(mse, mae, cIndex, rm2))
        #Record the cross validation result. We should only get the result from same validation round!
        crossValMSE.append(mse)
        crossValCI.append(cIndex)
        crossValMAE.append(mae)
        crossValRM2.append(rm2)
    # When cross validation finished, we record the result
    logger.LogInfoWithStr("SPLIT RESULT:", "MSE = {:.3f}({:.3f}), MAE = {:.3f}({:.3f}), CI = {:.3f}({:.3f}), R2 = {:.3f}({:.3f})".format(np.mean(crossValMSE), np.std(crossValMSE), np.mean(crossValMAE), np.std(crossValMAE), np.mean(crossValCI), np.std(crossValCI), np.mean(crossValRM2), np.std(crossValRM2)))
    return crossValMSE, crossValMAE, crossValCI, crossValRM2

       


def GenNFlodData(XD, XAdj, XDegree, XT, Y, label_row_inds, label_col_inds, PARA, dataset, nfolds):
    outer_train_sets = nfolds[0:5] # Get 0,1,2,3,4 Data as training and validation data
    test_set = nfolds[5] # The last one is fixed test data
    foldinds = len(outer_train_sets)
    test_sets = []
    val_sets = []
    train_sets = []
    
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)

    minMSEs, minMAEs, maxCIs, maxRM2s = RunNFoldExperiment(XD, XAdj, XDegree, XT, Y, label_row_inds, label_col_inds, PARA, train_sets, val_sets, test_sets)
    return minMSEs, minMAEs, maxCIs, maxRM2s

def prepare_interaction_pairs(XD, XAdj, XDegree, XT, Y, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        dataset[pair_ind].append(drug)

        adj = XAdj[rows[pair_ind]]
        dataset[pair_ind].append(adj)

        deg = XDegree[rows[pair_ind]]
        dataset[pair_ind].append(deg)

        target = XT[cols[pair_ind]]
        dataset[pair_ind].append(np.array(target, dtype=np.float32))

        dataset[pair_ind].append(np.array(Y[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        if pair_ind < len(rows) - 1:
            dataset.append([])
    return dataset

def RunExperiment(PARA, foldcount=6):  # 6-fold cross validation

    dataset = DataSet(PARA)
    PARA.proteinTokenSetSize = dataset.proteinTokenSetSize
    PARA.drugTokenSetSize = dataset.drugTokenSetSize

    XD, XAdj, XDegree, XT, Y = dataset.ParseData(PARA)

    XD = np.asarray(XD)
    XAdj = np.asarray(XAdj)
    XDegree = np.asarray(XDegree)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    PARA.drug_count = drugcount
    PARA.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)

    MSE = []
    MAE = []
    CI = []
    R2 = []


    for i in range(10):
        logger.LogInfoWithStr("Split {}".format(i),"==============================================================================================================================================")
        #Set rand num seed for 10 times random split
        random.seed(i + 1234)

        # Construct dataset
        if PARA.problemType == 1:
            nfolds = get_random_folds(len(label_row_inds),foldcount)
        if PARA.problemType == 2:
            nfolds = get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount)
        if PARA.problemType == 3:
            nfolds = get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)

        # Run N Flod Cross Validation Experiment
        minMSEs, minMAEs, maxCIs, maxRM2s = GenNFlodData(XD, XAdj, XDegree, XT, Y, label_row_inds,label_col_inds, PARA, dataset, nfolds)
        MSE.extend(minMSEs)
        MAE.extend(minMAEs)
        CI.extend(maxCIs)
        R2.extend(maxRM2s)

    logger.LogInfoWithStr("FINAL RESULT:", "==============================================================================================================================================")
    logger.LogInfoWithStr("FINAL RESULT:", "MSE = {:.3f}({:.3f}), MAE = {:.3f}({:.3f}), CI = {:.3f}({:.3f}), R2 = {:.3f}({:.3f})".format(np.mean(MSE), np.std(MSE), np.mean(MAE), np.std(MAE), np.mean(CI), np.std(CI), np.mean(R2), np.std(R2)))

if __name__ == "__main__":
    PARA = argparser()
    # Set Max Seq Len Response to Dataset Type
    if PARA.datasetPath == 'data/kiba/': #KIBA Dataset
        PARA.maxProteinLen = 1000
        PARA.maxDrugLen = 100
    else:
        PARA.maxProteinLen = 1200
        PARA.maxDrugLen = 85
    logger = LogModule()
    RunExperiment(PARA)