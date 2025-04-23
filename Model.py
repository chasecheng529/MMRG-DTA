import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from Arguments import argparser
from math import sqrt
PARA = argparser()
device = torch.device("cuda:" + str(PARA.GPU))
    
class MultiScaleWeightAdd(nn.Module):
    def __init__(self, numHidden):
        super(MultiScaleWeightAdd, self).__init__()
        self.w = nn.Parameter(torch.ones((3, numHidden * 3, PARA.maxDrugLen), dtype = torch.float32), requires_grad = True)
        self.epsilon = 0.0001
        self.conv = nn.Conv1d(numHidden * 3, numHidden * 3, 1, 1, 0)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
    def forward(self, x1, x2, x3):
        w = self.relu(self.w)
        weight = w / (torch.sum(w, dim = 0) + self.epsilon)
        return self.conv(self.swish(weight[0] * x1 + weight[1] * x2 + weight[2] * x3)).permute(0,2,1)

class GraphConvolution(nn.Module):                            
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, out_features, bias = False)

    def forward(self, input, adj, degree):
        adj = torch.bmm(degree, adj)
        A = torch.bmm(adj, degree)
        out = F.relu(self.fc1(torch.bmm(A, input)))
        return out
class MMRGEncoder(nn.Module):
    def __init__(self, hiddenNum):
        self.weightRes = 1
        self.weightFuse = 1
        super(MMRGEncoder, self).__init__()
        self.hiddenNum = hiddenNum
        self.dropout = 0.2
        
        self.mmrgGCN = GraphConvolution(hiddenNum * 3, hiddenNum * 3)
        self.multiscalefuse = MultiScaleWeightAdd(hiddenNum)
        self.Relu = nn.ReLU()
        self.Drop = nn.Dropout(0.2)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, hiddenNum * 3, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 3, 1, 3 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, hiddenNum * 3, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 5, 1, 5 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, hiddenNum * 3, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 7, 1, 7 // 2),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.out = nn.AdaptiveAvgPool1d(1)

        self.fuse = nn.Sequential(
            nn.Linear(hiddenNum * 9, hiddenNum * 3),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Linear(hiddenNum * 3, hiddenNum * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hiddenNum * 3, hiddenNum * 3),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1).to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x, adj, degree):
        x = x.permute(0,2,1)

        #Scale 3
        cnnx3 = self.conv2(x)

        #Scale 5
        cnnx5 = self.conv3(x)

        #Scale 7
        cnnx7 = self.conv4(x)

        #Fuse MultiScale Feature
        multiscaleFeature = self.multiscalefuse(cnnx3, cnnx5, cnnx7)
        graphconvFeature = (self.mmrgGCN(multiscaleFeature, adj, degree) + multiscaleFeature).permute(0,2,1)
    
        # build mu and var for VAE
        output = self.out(graphconvFeature)
        output = output.squeeze(2)
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2

class CNN(nn.Module):
    def __init__(self, num_filters, filterLen):
        super(CNN, self).__init__()
        self.hiddenNum = num_filters
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 128, out_channels = num_filters * 2, kernel_size = filterLen, stride = 1, padding = filterLen // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, filterLen, 1, filterLen // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, filterLen, 1, filterLen // 2),
        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1).to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        
        output = self.out(x)
        output = output.squeeze(2)
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2

class Decoder(nn.Module):
    def __init__(self, init_dim, num_filters, filterLen, size):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3 * (init_dim - 3 * (filterLen - 1))),
            nn.ReLU()
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 3, num_filters * 2, filterLen, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, filterLen, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, 128, filterLen, 1, 0),
            nn.ReLU(),
        )
        self.layer2 = nn.Linear(128, size)

    def forward(self, x, init_dim, num_filters, filterLen):
        x = self.layer(x)
        x = x.view(-1, num_filters * 3, init_dim - 3 * (filterLen - 1))
        x = self.convt(x)
        x = x.permute(0,2,1)
        x = self.layer2(x)
        return x

class RegBlock(nn.Module):
    def __init__(self, num_filters):
        super(RegBlock, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

        self.reg1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

        self.reg2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )


    def forward(self, A, B):
        A = self.reg1(A)
        B = self.reg2(B)
        x = torch.cat((A, B), 1)
        x = self.reg(x)
        return x
    
class MMRGNet(nn.Module):
    def __init__(self, PARA, proteinFilterLen, drugFilterLen):
        super(MMRGNet, self).__init__()
        self.drugFilterLen = drugFilterLen
        self.proteinFilterLen = proteinFilterLen
        self.embedding1 = nn.Embedding(PARA.drugTokenSetSize, 128, padding_idx = 0)
        self.embedding2 = nn.Embedding(PARA.proteinTokenSetSize, 128, padding_idx = 0)
        self.tiBlock = MMRGEncoder(PARA.numHidden)
        self.cnn = CNN(PARA.numHidden, proteinFilterLen)
        self.decoder1 = Decoder(PARA.maxDrugLen, PARA.numHidden, drugFilterLen, PARA.drugTokenSetSize)
        self.decoder2 = Decoder(PARA.maxProteinLen, PARA.numHidden, proteinFilterLen, PARA.proteinTokenSetSize)
        self.normalReg = RegBlock(PARA.numHidden)

    def forward(self, drugSmiles, adjMatrix, degree, proteinSeq, PARA):
        x_init = Variable(drugSmiles.long()).cuda(device)
        x = self.embedding1(x_init)
        adjMatrix = Variable(adjMatrix.float()).cuda(device)
        degree = Variable(degree.float()).cuda(device)
        y_init = Variable(proteinSeq.long()).cuda(device)
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        x, mu_x, logvar_x = self.tiBlock(x, adjMatrix, degree)
        prex = self.decoder1(x, PARA.maxDrugLen, PARA.numHidden, self.drugFilterLen)
        y, mu_y, logvar_y = self.cnn(y_embedding)
        prey = self.decoder2(y, PARA.maxProteinLen, PARA.numHidden, self.proteinFilterLen)
        affinity = self.normalReg(x, y).squeeze(1)
        return affinity, prex, prey, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y