import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from Arguments import argparser
from math import sqrt
PARA = argparser()
device = torch.device("cuda:" + str(PARA.GPU))

class TIBlock(nn.Module):
    def __init__(self, hiddenNum, filterLen):
        super(TIBlock, self).__init__()
        self.hiddenNum = hiddenNum
        self.dropout = 0.2
        self.gcn1 = GraphConvolution(128, hiddenNum * 2)
        self.gcn2 = GraphConvolution(hiddenNum * 1, hiddenNum * 4)
        self.gcn3 = GraphConvolution(hiddenNum * 2, hiddenNum * 6)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, hiddenNum * 2, filterLen, 1, filterLen // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(hiddenNum, hiddenNum * 4, filterLen, 1, filterLen // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(hiddenNum * 2, hiddenNum * 6, filterLen, 1, filterLen // 2),
        )

        self.out = nn.AdaptiveAvgPool1d(1)

        self.layer1 = nn.Sequential(
            nn.Linear(hiddenNum * 3, hiddenNum * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hiddenNum * 3, hiddenNum * 3),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
        self.weightCNN = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)
        self.weightGCN = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)

        self.SRK = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)
        self.SRQ = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)
        self.SRV = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)

        self.DRK = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)
        self.DRQ = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)
        self.DRV = nn.Conv1d(hiddenNum * 3, hiddenNum * 3, 1, 1, 0)

        self._norm_fact = 1/sqrt(hiddenNum * 3)
    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0,0.1).to(device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x, adj, degree):
        
        gcnx1 = self.gcn1(x, adj,degree)
        out, gate = gcnx1.split(int(gcnx1.size(2) / 2), 2)
        gcnx1 = out * self.sigmoid(gate)
        gcnx1 = F.dropout(gcnx1, self.dropout, training=self.training)

        gcnx2 = self.gcn2(gcnx1, adj,degree)
        out, gate = gcnx2.split(int(gcnx2.size(2) / 2), 2)
        gcnx2 = out * self.sigmoid(gate)
        gcnx2 = F.dropout(gcnx2, self.dropout, training=self.training)

        gcnx3 = self.gcn3(gcnx2, adj,degree)
        out, gate = gcnx3.split(int(gcnx3.size(2) / 2), 2)
        gcnx3 = out * self.sigmoid(gate)
        gcnx3 = F.dropout(gcnx3, self.dropout, training=self.training) # B * L * C
        gcnOut = gcnx3.permute(0,2,1)
        


        
        x = x.permute(0,2,1)
        cnnx = self.conv1(x)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv2(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv3(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate) # B * C * L
        cnnOut = cnnx

        
        SRQ = self.SRQ(cnnOut)
        SRK = self.SRK(cnnOut)
        SRV = self.SRV(cnnOut)

        DRQ = self.DRQ(gcnOut)
        DRK = self.DRK(gcnOut)
        DRV = self.DRV(gcnOut) #B * C * L

        attSR = nn.Softmax(dim=-1)(torch.bmm(SRQ.permute(0,2,1),DRK)) * self._norm_fact
        outSR = torch.bmm(attSR,SRV.permute(0,2,1))

        attDR = nn.Softmax(dim=-1)(torch.bmm(DRQ.permute(0,2,1),SRK)) * self._norm_fact
        outDR = torch.bmm(attDR,DRV.permute(0,2,1))
        
        
        


        #attentionScore = self.sigmoid(gcnx3)
        #attentionFeature = cnnx * attentionScore
        #attentionFeature = torch.matmul(self.weightCNN,cnnx) + torch.matmul(self.weightGCN,gcnx3)
        #attentionFeature = self.sigmoid(self.weightCNN(cnnx))*cnnx + self.sigmoid(self.weightGCN(gcnOut))*gcnOut
        #attentionFeature = torch.cat((outSR.permute(0,2,1),outDR.permute(0,2,1)),1)
        #attentionFeature = torch.cat((cnnOut,gcnOut),1)
        attentionFeature = outSR.permute(0,2,1) + outDR.permute(0,2,1)
        output = self.out(attentionFeature)
        output = output.squeeze(2)
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2
    
    def oldforward(self, x, adj, degree):
        x = x.permute(0,2,1)
        cnnx = self.conv1(x)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv2(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv3(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate) # B * Hidden * L

        output = self.out(cnnx)
        output = output.squeeze(2)
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2
    
class SeqTIBlock(nn.Module):
    def __init__(self, hiddenNum, filterLen):
        super(SeqTIBlock, self).__init__()
        self.hiddenNum = hiddenNum
        self.dropout = 0.2
        self.gcn1 = GraphConvolution(hiddenNum * 3, hiddenNum * 3)
        self.gcn2 = GraphConvolution(hiddenNum * 3, hiddenNum * 3)

        self.conv1 = nn.Sequential(
            nn.Conv1d(128, hiddenNum * 2, filterLen, 1, filterLen // 2),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(hiddenNum, hiddenNum * 4, filterLen, 1, filterLen // 2),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(hiddenNum * 2, hiddenNum * 6, filterLen, 1, filterLen // 2),
        )
        self.out = nn.AdaptiveAvgPool1d(1)

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
        cnnx = self.conv1(x)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv2(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate)
        cnnx = self.conv3(cnnx)
        out, gate = cnnx.split(int(cnnx.size(1) / 2), 1)
        cnnx = out * torch.sigmoid(gate) # B * C * L
        cnnOut = cnnx.permute(0,2,1)
        
        gcnx1 = self.gcn1(cnnOut, adj,degree)
        gcnx1 = F.dropout(gcnx1, self.dropout, training=self.training)


        gcnOut = gcnx1.permute(0,2,1)
        
        
        


        

        output = self.out(gcnOut)
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

class GraphConvolution(nn.Module):                            
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / torch.math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, degree):
        adj = torch.bmm(degree, adj)
        adj = torch.bmm(adj, degree)
        support = torch.matmul(input, self.weight)
        out = torch.bmm(adj, support)
        return out + self.bias

class TINet(nn.Module):
    def __init__(self, PARA):
        super(TINet, self).__init__()
        self.embedding1 = nn.Embedding(PARA.drugTokenSetSize, 128)
        self.embedding2 = nn.Embedding(PARA.proteinTokenSetSize, 128)
        self.tiBlock = SeqTIBlock(PARA.numHidden, PARA.drugFilterLen)
        self.drugcnn = CNN(PARA.numHidden, PARA.drugFilterLen)
        self.cnn = CNN(PARA.numHidden, PARA.proteinFilterLen)
        self.decoder1 = Decoder(PARA.maxDrugLen, PARA.numHidden, PARA.drugFilterLen, PARA.drugTokenSetSize)
        self.decoder2 = Decoder(PARA.maxProteinLen, PARA.numHidden, PARA.proteinFilterLen, PARA.proteinTokenSetSize)
        self.normalReg = NormalReg(PARA.numHidden)

    def forward(self, drugSmiles, adjMatrix, degree, proteinSeq, PARA):
        x_init = Variable(drugSmiles.long()).cuda(device)
        x = self.embedding1(x_init)
        adjMatrix = Variable(adjMatrix.float()).cuda(device)
        degree = Variable(degree.float()).cuda(device)
        y_init = Variable(proteinSeq.long()).cuda(device)
        y = self.embedding2(y_init)
        y_embedding = y.permute(0, 2, 1)
        x_embedding = x.permute(0,2,1)
        #x, mu_x, logvar_x = self.drugcnn(x_embedding)
        x, mu_x, logvar_x = self.tiBlock(x, adjMatrix, degree)
        prex = self.decoder1(x, PARA.maxDrugLen, PARA.numHidden, PARA.drugFilterLen)
        y, mu_y, logvar_y = self.cnn(y_embedding)
        prey = self.decoder2(y, PARA.maxProteinLen, PARA.numHidden, PARA.proteinFilterLen)
        affinity = self.normalReg(x, y).squeeze(1)
        return affinity, prex, prey, x_init, y_init, mu_x, logvar_x, mu_y, logvar_y
    
class NormalReg(nn.Module):
    def __init__(self, num_filters):
        super(NormalReg, self).__init__()
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