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
from Model import TINet
from AblationModel import AblationTINet
from sklearn.metrics import roc_auc_score, accuracy_score

from Log import LogModule
from Metrics import *
from DataHelper import *
from Arguments import argparser

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

def train(trainDataLoader, model, PARA):
    model.train()
    lossFunc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
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

            loss = affinityLoss + 10 ** PARA.lamda * (drugLoss + PARA.maxDrugLen / PARA.maxProteinLen * proteinLoss)
            loss.backward()
            optimizer.step()

            MSELossList.append(affinityLoss.item())
            drugLossList.append(drugLoss.item())
            targetLossList.append(proteinLoss.item())
            trainLossList.append(loss.item())

            t.set_postfix(TrainLoss=np.mean(trainLossList), MSE=np.mean(MSELossList))
        #logger.LogInfoWithArgs("Train:",MSE = np.mean(MSELossList), drugLoss=np.mean(drugLossList), targetLoss= np.mean(targetLossList), totalLoss=np.mean(trainLossList))
    return model

def test(testDataLoader, model, PARA):
    model.eval()
    lossFunc = nn.MSELoss()
    MAELossFunc = nn.L1Loss()
    affinities = []
    pre_affinities = []
    drugLoss = 0
    proteinLoss = 0
    with torch.no_grad():
        for i,(drugSMILES, drugAdj, drugDegree, proteinSeq, affinity) in enumerate(testDataLoader):
            drugSMILES = drugSMILES.cuda(device)
            proteinSeq = proteinSeq.cuda(device)
            drugDegree = drugDegree.cuda(device)

            if PARA.modelName == "TIVAE":
                pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, drugAdj, drugDegree, proteinSeq, PARA)
            else:
                pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, proteinSeq, PARA)
            
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()
            drugLoss += GenLoss(new_drug, drug, mu_drug, logvar_drug)
            proteinLoss += GenLoss(new_target, target, mu_target, logvar_target)

        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        MSELoss = lossFunc(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        MAELoss = MAELossFunc(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        cIndex = GetCIndex(affinities,pre_affinities)
        rm2 = GetRM2(affinities, pre_affinities)
    return cIndex, MSELoss, MAELoss, rm2

def testForPlot(testDataLoader, model, PARA):
    model.eval()
    

    affinities = []
    pre_affinities = []

    with torch.no_grad():
        for i,(drugSMILES, drugAdj, drugDegree, proteinSeq, affinity) in enumerate(testDataLoader):
            drugSMILES = drugSMILES.cuda(device)
            proteinSeq = proteinSeq.cuda(device)
            drugDegree = drugDegree.cuda(device)

            if PARA.modelName == "TIVAE":
                pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, drugAdj, drugDegree, proteinSeq, PARA)
            else:
                pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = model(drugSMILES, proteinSeq, PARA)
            
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()


        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        #Draw Image

# This function is used to run 6Fold Validation, each splited folds will be used 6 times!
def RunNFoldExperiment(XD, XAdj, XDegree, XT, Y, label_row_inds, label_col_inds, PARA, labeled_sets, val_sets, test_sets):
    crossValMSE = []
    crossValMAE = []
    crossValCI = []
    crossValRM2 = []

    AbcrossValMSE = []
    AbcrossValMAE = []
    AbcrossValCI = []
    AbcrossValRM2 = []

    for foldind in range(len(val_sets)):
        logger.LogInfoWithStr("FOLD:{}".format(foldind),"============================NEW FOLD=========================")
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
        val_loader = DataLoader(dataset = val_dataset, batch_size = PARA.batchSize) # Used for parameter search!

        model = TINet(PARA).cuda(device)
        model.apply(InitWeights)

        Abmodel = AblationTINet(PARA).cuda(device)
        Abmodel.apply(InitWeights)

        
        CIndexList = []
        MSEList = []
        MAEList = []
        RM2List = []

        AbCIndexList = []
        AbMSEList = []
        AbMAEList = []
        AbRM2List = []

        for epochind in range(PARA.numEpoch):
            model = train(train_loader, model, PARA)
            Abmodel = train(train_loader, Abmodel, PARA)


            cIndex, MSELoss, MAELoss, rm2 = test(test_loader, model, PARA) 
            AbcIndex, AbMSELoss, AbMAELoss, Abrm2 = test(test_loader, Abmodel, PARA) 

            CIndexList.append(cIndex)
            MSEList.append(MSELoss)
            MAEList.append(MAELoss)
            RM2List.append(rm2)

            AbCIndexList.append(AbcIndex)
            AbMSEList.append(AbMSELoss)
            AbMAEList.append(AbMAELoss)
            AbRM2List.append(Abrm2)

            logger.LogInfoWithStr('TEST', 'Epoch:{}, MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(epochind, MSELoss, MAELoss, cIndex, rm2))
            logger.LogInfoWithStr('AB TEST', 'Epoch:{}, MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(epochind, AbMSELoss, AbMAELoss, AbcIndex, Abrm2))
            '''
            if MSELoss <= min(MSEList):
                torch.save(model, '{}.pth'.format(PARA.gitNode[-5:]))
            '''
        logger.LogInfoWithStr('FOLD Result', 'MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(min(MSEList), min(MAEList), min(CIndexList), min(RM2List)))
        logger.LogInfoWithStr('AB FOLD Result', 'MSE:{:.3f}, MAE:{:.3f}, Cindex:{:.3f}, RM2:{:.3f}'.format(min(AbMSEList), min(AbMAEList), min(AbCIndexList), min(AbRM2List)))

        #Record the cross validation result. We should only get the result from same validation round!
        crossValMSE.append(min(MSEList))
        crossValCI.append(min(CIndexList))
        crossValMAE.append(min(MAEList))
        crossValRM2.append(min(RM2List))

        AbcrossValMSE.append(min(AbMSEList))
        AbcrossValCI.append(min(AbCIndexList))
        AbcrossValMAE.append(min(AbMAEList))
        AbcrossValRM2.append(min(AbRM2List))
        '''
        # When training finished, record the best result
        loss_func = nn.MSELoss()
        mae_loss_func = nn.L1Loss()
        affinities = []
        pre_affinities = []
        model=torch.load('{}.pth'.format(PARA.gitNode[-5:]))
        model.eval()
        for drugSMILES, drugAdj, drugDegree, proteinSeq, affinity in test_loader:
            pre_affinity, _, _, _, _, _, _, _, _ = model(drugSMILES, drugAdj, drugDegree, proteinSeq, PARA)
            pre_affinities += pre_affinity.cpu().detach().numpy().tolist()
            affinities += affinity.cpu().detach().numpy().tolist()

        pre_affinities = np.array(pre_affinities)
        affinities = np.array(affinities)
        MSELoss = loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        MAELoss = mae_loss_func(torch.Tensor(pre_affinities), torch.Tensor(affinities))
        cIndex = GetCIndex(affinities,pre_affinities)
        rm2 = GetRM2(affinities, pre_affinities)
        '''

    # When cross validation finished, we record the result
    logger.LogInfoWithStr("FOLD FINAL RESULT:", "MSE = {:.3f}({:.3f}), MAE = {:.3f}({:.3f}), CI = {:.3f}({:.3f}), R2 = {:.3f}({:.3f})".format(np.mean(crossValMSE), np.std(crossValMSE), np.mean(crossValMAE), np.std(crossValMAE), np.mean(crossValCI), np.std(crossValCI), np.mean(crossValRM2), np.std(crossValRM2)))
    logger.LogInfoWithStr("AB FOLD FINAL RESULT:", "MSE = {:.3f}({:.3f}), MAE = {:.3f}({:.3f}), CI = {:.3f}({:.3f}), R2 = {:.3f}({:.3f})".format(np.mean(AbcrossValMSE), np.std(AbcrossValMSE), np.mean(AbcrossValMAE), np.std(AbcrossValMAE), np.mean(AbcrossValCI), np.std(AbcrossValCI), np.mean(AbcrossValRM2), np.std(AbcrossValRM2)))

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
        random.seed(i + 1000)

        # Construct dataset
        if PARA.problemType == 1:
            nfolds = get_random_folds(len(label_row_inds),foldcount)
        if PARA.problemType == 2:
            nfolds = get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount)
        if PARA.problemType == 3:
            nfolds = get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)

        # Run N Flod Cross Validation Experiment
        minMSE, minMAE, maxCI, maxRM2 = GenNFlodData(XD, XAdj, XDegree, XT, Y, label_row_inds,label_col_inds, PARA, dataset, nfolds)
        MSE.extend(minMSE)
        MAE.extend(minMAE)
        CI.extend(maxCI)
        R2.extend(maxRM2)
    logger.LogInfoWithStr("FINAL RESULT:", "==============================================================================================================================================")
    logger.LogInfoWithStr("FINAL RESULT:", "MSE = {:.3f}({:.3f}), MAE = {:.3f}({:.3f}), CI = {:.3f}({:.3f}), R2 = {:.3f}({:.3f})".format(np.mean(MSE), np.std(MSE), np.mean(MAE), np.std(MAE), np.mean(CI), np.std(CI), np.mean(R2), np.std(R2)))
    logger.LogInfoWithArgs("Key Settings:", Model = PARA.modelName, Problem = PARA.problemType, PFL = PARA.proteinFilterLen, DFL = PARA.drugFilterLen, Lambda = PARA.lamda, PL = PARA.maxProteinLen, DL = PARA.maxDrugLen, NumHidden = PARA.numHidden)

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