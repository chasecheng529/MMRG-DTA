import numpy as np
from rdkit import Chem
from torch.utils import data
import json
from collections import OrderedDict
import pandas as pd
import math
from Arguments import argparser

PARA = argparser()

# This part is for tokenize SMILES
MIXATOM = ['Cl', 'Br']

SMILESATOMSET = ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P' 'I']

SMILESTOKENSET = {"#": 1,"=": 2,"(": 3, ")": 4, "+": 5,
                    "-": 6, ".": 7, "0": 8, "1": 9, "2": 10,
                    "3": 11, "4": 12, "5": 13, "6": 14, "7": 15,
                    "8": 16, "9": 17, "C": 18, "N":19, "O":20,
                    "F":21, "P":22, "S":23, "Cl":24, "Br":25,
                    "I":26, "[":27, "]":28}

SMILESTOKENSETLEN = 29

# This part is for tokenize Protein Seq
PROTEINCHARSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5,
                  "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
                  "M": 11, "L": 12,"O": 13, "N": 14, "Q": 15,
                  "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
                  "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

PROTEINCHARSETLEN = 26

def TokenrizeSmiles(smilesStr, smilesLen):
    smilesToken = np.zeros((smilesLen))
    atomIdxList = []
    tokenIdx = 0
    symbolIdx = 0
    atomIdx = 0
    while symbolIdx < len(smilesStr):
        if tokenIdx >= smilesLen:
            break
        if smilesStr[symbolIdx : symbolIdx + 2 if symbolIdx < len(smilesStr)-1 else symbolIdx] in MIXATOM:
            atomIdxList.append(tokenIdx)
            smilesToken[tokenIdx] = (SMILESTOKENSET[smilesStr[symbolIdx : symbolIdx + 2]])
            tokenIdx += 1
            atomIdx += 1
            symbolIdx = symbolIdx + 2
        else:
            smilesToken[tokenIdx] = (SMILESTOKENSET[smilesStr[symbolIdx]])
            if smilesStr[symbolIdx] in SMILESATOMSET:
                atomIdxList.append(tokenIdx)
                atomIdx += 1
            tokenIdx += 1
            symbolIdx = symbolIdx + 1
    return smilesToken, atomIdxList

def ProteinToVec(proteinSeq): 
    X = np.zeros(1000)
    for i, char in enumerate(proteinSeq[:1000]):
        X[i] = PROTEINCHARSET[char]
    return X  

def GetRemoveList(listName, length):
    removelist = []
    # Davis  SMILES:85 protein:1200
    # KIBA    SMILES:100   protein:1000
    for i, x in enumerate(listName):
        if len(x) > length:
            removelist.append(i)
    return removelist

def RemoveFromList(listName, removeList):
    a_index = [i for i in range(len(listName))]
    a_index = set(a_index)
    b_index = set(removeList)
    index = list(a_index - b_index)
    a = [listName[i] for i in index]
    return a

def DataFrameRemove(dataframe, removelist, axis):
    if axis == 0:
        new_df = dataframe.drop(removelist)
        new_df = new_df.reset_index(drop=True)
    if axis == 1:
        new_df = dataframe.drop(removelist, axis=1)
        new_df.columns = range(new_df.shape[1])
    return new_df

def Orderdict2List(dict):
    x = []
    for d in dict.keys():
        x.append(dict[d])
    return x

class DataSet(object):
    def __init__(self, PARA):
        self.proteinTokenSet = PROTEINCHARSET
        self.proteinTokenSetSize = PROTEINCHARSETLEN

        self.drugTokenSet = SMILESTOKENSET
        self.drugTokenSetSize = SMILESTOKENSETLEN


    def ParseData(self, PARA):
        data_path = PARA.datasetPath
        drugSMILES = json.load(open(data_path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
        proteinSeqs = json.load(open(data_path + "proteins.txt"), object_pairs_hook=OrderedDict)
        drugSMILES = Orderdict2List(drugSMILES)
        proteinSeqs = Orderdict2List(proteinSeqs)

        if 'davis' in data_path:
            affinities = pd.read_csv(data_path + 'drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt',
                                     sep='\s+',
                                     header=None, encoding='latin1')
            affinities = -(np.log10(affinities / (math.pow(10, 9))))
        else:
            affinities = pd.read_csv(data_path + 'kiba_binding_affinity_v2.txt', sep='\s+', header=None,
                                     encoding='latin1')
            
            drugRemove = GetRemoveList(drugSMILES, 90)
            proteinRemove = GetRemoveList(proteinSeqs, 1365)

            drugSMILES = RemoveFromList(drugSMILES, drugRemove)
            proteinSeqs = RemoveFromList(proteinSeqs, proteinRemove)

            affinities = DataFrameRemove(affinities, drugRemove, 0)
            affinities = DataFrameRemove(affinities, proteinRemove, 1)
            

        XDrugSmile = []
        XAdj = []
        XDegree = []
        XProtein = []
        for drug in drugSMILES:
            smilesToken, adjMatrix = SMILES2SeqAndAdj(drug)
            degreeMatrix = GetDegreeMatrix(adjMatrix)
            XDrugSmile.append(smilesToken)
            XAdj.append(adjMatrix)
            XDegree.append(degreeMatrix)
        for protein in proteinSeqs:
            XProtein.append(TokenrizeProtein(protein))
        return XDrugSmile, XAdj, XDegree, XProtein, np.array(affinities)

# For dataset do not contain bond information data, you need use this function to create adjuntion matrix
def SMILES2SeqAndAdj(smile, smilesLen = PARA.maxDrugLen):
    smilesToken, atomIdxList = TokenrizeSmiles(smile, smilesLen)
    adjMatrix = np.zeros((smilesLen,smilesLen))
    mol = Chem.MolFromSmiles(smile)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start >= len(atomIdxList) or end >= len(atomIdxList):
            continue
        adjMatrix[int(atomIdxList[start])][int(atomIdxList[end])] = 1
        adjMatrix[int(atomIdxList[end])][int(atomIdxList[start])] = 1
    for tokenidx in range(len(smilesToken)):
        adjMatrix[tokenidx][tokenidx] = 1
    
    return smilesToken, adjMatrix

def TokenrizeProtein(protein, proteinLen = PARA.maxProteinLen):
    proteinToken = np.zeros((proteinLen))
    for i, token in enumerate(protein[:proteinLen]):
        proteinToken[i] = PROTEINCHARSET[token]
    return proteinToken

    
def GetDegreeMatrix(adjMatrix, smilesLen = PARA.maxDrugLen):
    degreeMatrix = np.zeros((smilesLen,smilesLen))
    for i in range(smilesLen):
        degree = sum(adjMatrix[i])
        degreeMatrix[i][i] = 0 if degree == 0 else np.power(degree, -0.5)
    return degreeMatrix