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
MIXATOM = ['Cl', 'Br', 'Na', 'Si', 'Ca']

MIXSYMBOL = ["@@"]

SMILESATOMSET = ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'Si', 'I', 'B', 'Na', 'H', 'Ca']

ATOMSET = ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'Si', 'I', 'B', 'Na', 'H', 'Ca']
'''
SMILESTOKENSET = {'7':1, '4':2, '@':3, 'C':4, '(':5, '+':6, '9':7,
                '.':8, '2':9, '-':10, '/':11, 'O':12, 'Br':13,
                'H':14, 'N':15, ']':16, '#':17, 'I':18, 'S':19, '=':20,
                '\\':21, 'F':22, 'P':23, '6':24, '8':25,
                'Cl':26, ')':27, '5':28, '[':29, '1':30, '3':31,'@@':32}

SMILESTOKENSETLEN = 33
'''
SMILESTOKENSET = {'7':1, '4':2, '@':3, 'C':4, '(':5, '+':6, '9':7,
                '.':8, '2':9, '-':10, '/':11, 'O':12, 'Br':13,
                'H':14, 'N':15, ']':16, '#':17, 'I':18, 'S':19, '=':20,
                '\\':21, 'F':22, 'P':23, '6':24, '8':25,
                'Cl':26, ')':27, '5':28, '[':29, '1':30, '3':31,
                'Na':32, 'Si':33, 'B':34, '0':35, '%':36, '@@':37,
                'Ca':38, 'c':39, 'o':40, 's':41, 'n':42}

SMILESTOKENSETLEN = 43


# This part is for tokenize Protein Seq
PROTEINCHARSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5,
                  "G": 6, "F": 7, "I": 8, "H": 9, "K": 10,
                  "M": 11, "L": 12,"O": 13, "N": 14, "Q": 15,
                  "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
                  "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

PROTEINCHARSETLEN = 26

def TokenrizeSmiles(smilesStr, smilesLen):
    smilesToken = np.zeros((smilesLen))
    atomIdxList = np.zeros((smilesLen))
    tokenIdx = 0
    symbolIdx = 0
    atomIdx = 0
    while symbolIdx < len(smilesStr):
        if symbolIdx >= smilesLen:
            break
        if smilesStr[symbolIdx : symbolIdx + 2 if symbolIdx < len(smilesStr)-1 else symbolIdx] in MIXATOM:
            atomIdxList[atomIdx] = tokenIdx
            smilesToken[tokenIdx] = (SMILESTOKENSET[smilesStr[symbolIdx : symbolIdx + 2]])
            tokenIdx += 1
            atomIdx += 1
            symbolIdx = symbolIdx + 2
        elif smilesStr[symbolIdx : symbolIdx + 2 if symbolIdx < len(smilesStr)-1 else symbolIdx] in MIXSYMBOL:
            smilesToken[tokenIdx] = (SMILESTOKENSET[smilesStr[symbolIdx : symbolIdx + 2]])
            tokenIdx += 1
            symbolIdx = symbolIdx + 2
        else:
            smilesToken[tokenIdx] = (SMILESTOKENSET[smilesStr[symbolIdx]])
            if smilesStr[symbolIdx] in SMILESATOMSET:
                atomIdxList[atomIdx] = tokenIdx
                atomIdx += 1
            tokenIdx += 1
            symbolIdx = symbolIdx + 1
    return smilesToken, atomIdxList

# If dataset contains BondInfo data, use this to parse it and generate adjunction matrix
def ParseBondInfo(bondInfo):
    bondPairList = bondInfo.split(',')
    parsedBondInfo = []
    for eachBondPair in bondPairList:
        startEndPair = eachBondPair.split('-')
        parsedBondInfo.append((startEndPair[0], startEndPair[1]))
    return parsedBondInfo

def GetAdjMatrix(bondInfo, atomIdxList):
    adjMatrix = np.zeros((100,100))
    parsedBondInfo = ParseBondInfo(bondInfo)
    for bond in parsedBondInfo:
        start, end = int(bond[0]), int(bond[1])
        if start >= 100 or end >= 100: # In normal case, the lenth of adjuntion matrix can not be bigger than 100
            continue
        adjMatrix[int(atomIdxList[start])][int(atomIdxList[end])] = 1
        adjMatrix[int(atomIdxList[end])][int(atomIdxList[start])] = 1
    return adjMatrix

def Drug2SeqAndAdj(smiles, bondInfo):
    smilesToken, atomIdxList = TokenrizeSmiles(smiles, PARA.maxDrugLen)
    adjMatrix = GetAdjMatrix(bondInfo, atomIdxList)
    return smilesToken, adjMatrix

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



# This part is the interface of the module
class DrugDataset(data.Dataset):
    def __init__(self):
        self.smilePath = "data/SMILES/CID-SMILES-LEN-VALID-1M.txt"
        self.bondPath = "data/SMILES/CID-ADJ-VALID-1M.txt"
        smileFile = open(self.smilePath, 'r')
        bondFile = open(self.bondPath,'r')
        self.smilesData = smileFile.readlines()
        self.bondData = bondFile.readlines()
        smileFile.close()
        bondFile.close()
    def __getitem__(self, index):
        smiles = self.smilesData[index-1]
        bondInfo = self.bondData[index-1]
        smilesSeq, adjMatrix = Drug2SeqAndAdj(smiles.rstrip('\n'), bondInfo.rstrip('\n'))
        return smilesSeq, adjMatrix
    def __len__(self):
        return len(self.smilesData)
    
class ProteinDataset(data.Dataset):
    def __init__(self):
        self.proteinPath = "data/ProteinSeq/ProteinSeq-LEN1000.txt"
        proteinFile = open(self.proteinPath, 'r')
        self.proteinData = proteinFile.readlines()
        proteinFile.close()
    def __getitem__(self, index):
        protein = self.proteinData[index-1]
        proteinVector = ProteinToVec(protein.rstrip('\n'))
        return proteinVector
    def __len__(self):
        return len(self.proteinData)

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
        if PARA.modelName == "TIVAE":
            self.drugTokenSetSize = SMILESTOKENSETLEN
        else:
            self.drugTokenSetSize = 64


    def ParseData(self, PARA):
        data_path = PARA.datasetPath
        drugSMILES = json.load(open(data_path + "ligands_iso.txt"), object_pairs_hook=OrderedDict)
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
            if PARA.modelName == 'TIVAE':
                smilesToken, adjMatrix = SMILES2SeqAndAdj(drug)
            else:
                smilesToken, adjMatrix = TokenrizeDrugForOldCoding(drug)
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
    I = np.identity(smilesLen)
    neighborMatrix = np.zeros((smilesLen,smilesLen))
    np.fill_diagonal(neighborMatrix[1:],1)
    np.fill_diagonal(neighborMatrix[:,1:],1)
    mol = Chem.MolFromSmiles(smile)
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start >= smilesLen or end >= smilesLen: # In normal case, the lenth of adjuntion matrix can not be bigger than 100
            continue
        if atomIdxList[start] >= smilesLen or atomIdxList[end] >= smilesLen: # In normal case, the lenth of adjuntion matrix can not be bigger than 100
            continue
        adjMatrix[int(atomIdxList[start])][int(atomIdxList[end])] = 1
        adjMatrix[int(atomIdxList[end])][int(atomIdxList[start])] = 1
    adjMatrix = adjMatrix + I
    '''
    for atomIdx in atomIdxList:
        if atomIdx >= smilesLen:
            continue
        adjMatrix[int(atomIdx)][int(atomIdx)] = 1
    
    for tokenidx in range(len(smilesToken)):
        if smilesToken[tokenidx] == 0:
            break
        adjMatrix[tokenidx][tokenidx] = 1
    '''
    
    
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


ODLATOMSET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def TokenrizeDrugForOldCoding(smile, smilesLen = PARA.maxDrugLen):
    drugtoken = np.zeros((smilesLen))
    adj = np.zeros((smilesLen, smilesLen))
    for i, token in enumerate(smile[:smilesLen]):
        drugtoken[i] = ODLATOMSET[token]
    return drugtoken, adj