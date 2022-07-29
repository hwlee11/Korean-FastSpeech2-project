import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot


class MultiHeadAttention(nn.Module):
    def __init__(self,dDim=512,numOfHeads=8):
        super().__init__()
        self.attentionList = nn.ModuleList([SelfAttention(dDim) for i in range(numOfHeads)])

    def forward(self,x):
        for i in self.attentionList:
            print(i)


class SelfAttention(nn.Module):
    def __init__(self,dDim=512,numOfHeads=8):
        super().__init__()

        self.dDim = dDim
        self.kDim = dDim
        self.numOfHeads = numOfHeads
        
        self.wQ = nn.Linear(dDim,dDim*numOfHeads,bias=False)
        self.wK = nn.Linear(dDim,dDim*numOfHeads,bias=False)
        self.wV = nn.Linear(dDim,dDim*numOfHeads,bias=False)
        self.wH = nn.Linear(dDim*numOfHeads,dDim,bias=False)
        '''
        self.wQ = nn.Linear(dDim,dDim)
        self.wK = nn.Linear(dDim,dDim)
        self.wV = nn.Linear(dDim,dDim)
        '''
        self.softMax = nn.Softmax(dim=3)

    def forward(self,x,attentionMask=None,h=None):

        t = x.size()[1]
        q = self.wQ(x)
        if h == None:
            k = self.wK(x)
            v = self.wV(x)
        else:
            k = self.wK(h)
            v = self.wV(h)
        #print('v',v)

        q = self.splitHead(q,q.size()[1])
        k = self.splitHead(k,k.size()[1])
        v = self.splitHead(v,v.size()[1])
        if attentionMask == None:
            a = torch.matmul(q,torch.transpose(k,2,3))/math.sqrt(self.kDim)
            a = self.softMax(a)
            h = torch.matmul(a,v)
        else:
            a = torch.matmul(q,torch.transpose(k,2,3))/math.sqrt(self.kDim)
            attentionMask=self.getMaskTensor(a,attentionMask)
            #print(attentionMask,attentionMask.size())
            a.masked_fill_(attentionMask,-1e38)
            a = self.softMax(a)
            #print(a,v)
            h = torch.matmul(a,v)
            #print('H tensro',h)
        #h = h.transpose(1,2).contiguous().view(-1,t,self.dDim*self.numOfHeads)
        h = h.transpose(1,2).reshape(-1,t,self.dDim*self.numOfHeads).contiguous()
        #h = h.transpose(1,2).view(-1,t,self.dDim*self.numOfHeads).contiguous()
        h = self.wH(h)

        return h

    def printWeight(self):
        print(self.wQ.weight)
    
    def splitHead(self,w,t):
        return  w.view(-1,t,self.numOfHeads,self.dDim).transpose(1,2).contiguous()

    def getMaskTensor(self,a,lengths):
        numBatchs,maxLengths = lengths.size()
        mask = torch.zeros_like(a,dtype=torch.bool)
        for i in range(numBatchs):
            a = lengths[i].unsqueeze(0).expand(maxLengths,maxLengths)
            b = lengths[i].unsqueeze(0).expand(maxLengths,maxLengths)
            c = torch.logical_or(a.T,b)
            for j in range(self.numOfHeads):
                mask[i][j].data.copy_(c)

        return mask


class Predictor(nn.Module):
    def __init__(self,numOfPredictorLayer,dDim,numClass,kernelSize=3,dropout=0.5,bDuration=False):
        super().__init__()

        self.dDim = dDim
        self.bDuration = bDuration
        self.numOfPredictorLayer = numOfPredictorLayer
        self.convlist = nn.ModuleList([nn.Conv1d(dDim,dDim,kernelSize,stride=1,padding=kernelSize//2) for i in range(numOfPredictorLayer)])
        self.layerNormList = nn.ModuleList([nn.LayerNorm(dDim) for i in range(numOfPredictorLayer)])
        self.linear = nn.Linear(dDim,numClass)   # extra linear layer to output a scalar arXiv:1905.09263
        self.ReLU = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.loss = nn.MSELoss()

    def forward(self,x,lab=None,lengths=None):

        for i in range(self.numOfPredictorLayer):
            x = self.ReLU(self.convlist[i](x.transpose(1,2)).transpose(1,2))
            x = self.drop(self.layerNormList[i](x))

        if self.bDuration:
            d = self.linear(x).squeeze(-1)
            #mask = self.getMaskTensor(d,lengths)
        else:
            d = self.linear(x)

        if lab == None:
            return d
        #d.masked_fill_(lengths,0.0)
        '''
        loss = self.loss(d,lab)
        #if self.bDuration:
        #    print(loss)
        return d,loss
        '''
        return d

class VarianceAdaptor(nn.Module):
    def __init__(self,numOfDurationPredictorLayer,
            numOfPitchPredictorLayer,
            numOfEnergyPredictorLayer,
            kernelSize=3,
            dDim=256,dropout=0.5,bGenerating=False):
        super().__init__()
        self.dDim = dDim
        self.bGenerating = bGenerating
        self.durationPredictor = Predictor(numOfDurationPredictorLayer,dDim,1,kernelSize=kernelSize,dropout=dropout,bDuration=True)
        self.pitchPredictor = Predictor(numOfPitchPredictorLayer,dDim,dDim,kernelSize=kernelSize,dropout=dropout)
        self.energyPredictor = Predictor(numOfEnergyPredictorLayer,dDim,dDim,kernelSize=kernelSize,dropout=dropout)

        self.pitchEmdedder = nn.Embedding(dDim,dDim)
        self.energyEmdedder = nn.Embedding(dDim,dDim)

    def forward(self,x,mfAlign=None,pitch=None,energy=None,lengths=None,alpha=1.0,pitchAlpha=1.0,energyAlpha=1.0):


        if self.bGenerating == False :
            """
            # phoneme duration
            d,dLoss = self.durationPredictor(x.detach(),mfAlign,lengths)
            d = torch.round(alpha*d)
            hiddenSeq,hiddenSeqLength = self.lengthRegulator(x,mfAlign)
            pitchEmbeddingSeq = self.pitchEmbedding(pitch,pitchAlpha)
            energyEmbeddingSeq = self.energyEmbedding(energy,energyAlpha)
            p,pLoss = self.pitchPredictor(hiddenSeq.detach(),pitchEmbeddingSeq)
            e,eLoss = self.energyPredictor(hiddenSeq.detach(),energyEmbeddingSeq)
            hiddenSeq = hiddenSeq + pitchEmbeddingSeq + energyEmbeddingSeq
            return hiddenSeq,hiddenSeqLength,dLoss,pLoss,eLoss
            """
            d = self.durationPredictor(x.detach(),mfAlign,lengths)
            if alpha != 1.0:
                d = torch.round(alpha*d)
            hiddenSeq,hiddenSeqLength = self.lengthRegulator(x,mfAlign)
            pitchEmbeddingSeq = self.pitchEmbedding(pitch,pitchAlpha)
            energyEmbeddingSeq = self.energyEmbedding(energy,energyAlpha)
            p = self.pitchPredictor(hiddenSeq.detach(),pitchEmbeddingSeq)
            e = self.energyPredictor(hiddenSeq.detach(),energyEmbeddingSeq)
            hiddenSeq = hiddenSeq + pitchEmbeddingSeq + energyEmbeddingSeq
            return hiddenSeq,hiddenSeqLength,d,p,e,pitchEmbeddingSeq,energyEmbeddingSeq

        else:
            d = self.durationPredictor(x.detach())
            d = torch.round(alpha*d)
            hiddenSeq,hiddenSeqLength = self.lengthRegulator(x,d)
            pitchEmbeddingSeq = self.pitchPredictor(hiddenSeq)
            energyEmbeddingSeq = self.energyPredictor(hiddenSeq)
            hiddenSeq = hiddenSeq + pitchEmbeddingSeq + energyEmbeddingSeq
            return hiddenSeq,hiddenSeqLength

    def lengthRegulator(self,x,dTensor,alpha=1.0):

        maxLengthOfHiddenSeq = int(dTensor.sum(1).max().item())
        numBatch = 0
        hiddenSeqList = list()
        lengthList = list()
        for i in x: # batch
            idx = 0
            tmp = list()
            for j in i: # time
                d = dTensor[numBatch][idx].item()
                tmp.append(j.expand(max(int(d),0),self.dDim))
                idx +=1
            hiddenSeq = torch.cat(tmp,0)
            seqLength = hiddenSeq.size()[0]
            lengthList.append(seqLength)
            if seqLength < maxLengthOfHiddenSeq:
                hiddenSeq = F.pad(hiddenSeq,(0,0,0,maxLengthOfHiddenSeq-seqLength),"constant")
            hiddenSeqList.append(hiddenSeq.unsqueeze(0))
            numBatch += 1

        hiddenSeq = torch.cat(hiddenSeqList,0)
        lengthTensor = torch.tensor(lengthList)
        return hiddenSeq,lengthTensor

    def pitchEmbedding(self,pitch,alpha=1.0):
        pitch = (alpha*pitch).to(torch.long)
        pitchEmbedding = self.pitchEmdedder(pitch)
        return pitchEmbedding

    def energyEmbedding(self,energy,alpha=1.0):
        energy = (alpha*energy).to(torch.long)
        energyEmbedding = self.energyEmdedder(energy)
        return energyEmbedding

    def muEncoding(x,mu=255):

        sgn = torch.sign(x)
        quantaizedF0 = sgn * ( torch.log(1+mu*np.absolute(f0) )/torch.log(1+mu) )
        quantaizedF0 = ((mu+1)*quantaizedF0/torch.log(mu+torch.ones_like(quantaizedF0))).to(int)

        return quantaizedF0

#
class FFTBlock(nn.Module):
    def __init__(self,numOfLayers,dDim,numOfHeads,innerLayerDim,kernelSize,dropoutProbability=0.5):
        super().__init__()

        self.numOfLayers = numOfLayers
        self.dDim = dDim
        self.numOfHeads = numOfHeads
        self.selfAttentionList = nn.ModuleList([SelfAttention(dDim,numOfHeads) for i in range(numOfLayers)])
        #self.firstMLPList = nn.ModuleList([nn.Linear(dDim,dDim) for i in range(numOfLayers)])
        #self.secondMLPList = nn.ModuleList([nn.Linear(dDim,dDim) for i in range(numOfLayers)])
        self.firstMLPList = nn.ModuleList([nn.Conv1d(dDim,innerLayerDim,kernelSize,stride=1,padding=kernelSize//2) for i in range(numOfLayers)])
        self.secondMLPList = nn.ModuleList([nn.Conv1d(innerLayerDim,dDim,kernelSize,stride=1,padding=kernelSize//2) for i in range(numOfLayers)])
        self.firstLayerNormList = nn.ModuleList([nn.LayerNorm(dDim) for i in range(numOfLayers)])
        self.secondLayerNormList = nn.ModuleList([nn.LayerNorm(dDim) for i in range(numOfLayers)]) 
        self.ReLU = nn.ReLU()
        self.drop = nn.Dropout(p=dropoutProbability)

    def printWeight(self,):
        print(self.selfAttentionList[0].printWeight())

    def forward(self,x,mask=None):
        for i in range(self.numOfLayers):
            #Multi head attention
            if mask == None:
                x1 = self.selfAttentionList[i](x)
            else:
                x1 = self.selfAttentionList[i](x,mask)
            x1 = self.drop(x1)
            x = self.firstLayerNormList[i](x1+x)

            #FFN
            #if mask != None:
            #    x.masked_fill_(mask.unsqueeze(-1).expand(x.size()[0],x.size()[1],self.dDim),0)
            x1 = self.secondMLPList[i](self.ReLU(self.firstMLPList[i](x.transpose(1,2)))).transpose(1,2)
            x1 = self.drop(x1)
            x = self.secondLayerNormList[i](x1+x)

        return x

class FastSpeech2(nn.Module):
    def __init__(self,
            numOfEncoderBlocks=4,
            encoderFftInnerDim=1024,
            encoderKernelSize=9,
            encoderNumOfHeads=2,
            numOfDecoderBlocks=4,
            decoderFftInnerDim=1024,
            decoderKernelSize=9,
            decoderNumOfHeads=2,
            numOfPhones=69,
            phoneEmbeddingDim=256,
            encoderHiddenDim=256,
            decoderHiddenDim=256,
            outputSpectrogramDim=128,
            numOfDurationPredictorLayer=2,
            numOfPitchPredictorLayer=2,
            numOfEnergyPredictorLayer=2,
            predictorKernelSize=3,
            lossFunction='MAE',
            predictorLossFunction='MSE',
            fftDropOut=0.1,
            predictorDropOut=0.5,
            bGenerating=False):
        super().__init__()

        self.bGenerating = bGenerating
        #self.charEmbeddingLayer = nn.Linear(numOfPhones,phoneEmbeddingDim,bias=False)
        self.charEmbeddingLayer = nn.Embedding(numOfPhones,phoneEmbeddingDim)
        #encoder
        self.encoderFFTBlocks = FFTBlock(numOfLayers=numOfEncoderBlocks,dDim=encoderHiddenDim,numOfHeads=encoderNumOfHeads,innerLayerDim=encoderFftInnerDim,kernelSize=encoderKernelSize,dropoutProbability=fftDropOut)
        #variance adaptor    forward(self,x,mfAlign,pitch,energy,lengths=None,alpha=1.0,pitchAlpha=1.0,energyAlpha=1.0)
        self.varianceAdaptor = VarianceAdaptor(numOfDurationPredictorLayer=numOfDurationPredictorLayer,
                numOfPitchPredictorLayer=numOfPitchPredictorLayer,
                numOfEnergyPredictorLayer=numOfEnergyPredictorLayer,
                kernelSize=predictorKernelSize,dDim=phoneEmbeddingDim,bGenerating=bGenerating)
        #decoder
        self.decoderFFTBlocks = FFTBlock(numOfLayers=numOfDecoderBlocks,dDim=decoderHiddenDim,numOfHeads=decoderNumOfHeads,innerLayerDim=decoderFftInnerDim,kernelSize=decoderKernelSize,dropoutProbability=fftDropOut)
        self.outputSpectrogramDim = outputSpectrogramDim
        self.decoderHiddenDim = decoderHiddenDim
        self.linear = nn.Linear(decoderHiddenDim,outputSpectrogramDim)
        #self.dDim = dDim
        if lossFunction == 'MAE':
            self.loss = nn.L1Loss()
            #self.loss = nn.L1Loss(reduction='sum')
        elif lossFunction == 'MSE':
            #self.loss = nn.MSELoss(reduction='sum')
            self.loss = nn.MSELoss(reduction='sum')
        self.predictorLoss = nn.MSELoss()

    def forward(self,x,spectrogram=None,mfAlign=None,lengths=None,pitch=None,energy=None,device='cpu'):

        x = self.charEmbeddingLayer(x)
        charPositionEmbeddingTensor = self.positionEmbedding(x,lengths).to(device)#.expand(x.size()[0],-1,-1)
        x += charPositionEmbeddingTensor
        lengthTensor = self.getLengthTensor(lengths)#.to(device)
        #x.masked_fill_(lengthTensor.unsqueeze(-1).expand(x.size()[0],x.size()[1],self.dDim),0)
        #masks = self.getMaskTensor(lengths)
        x = self.encoderFFTBlocks(x,lengthTensor)
        
        #hiddenSeq,hiddenLength, dLoss = self.varianceAdaptor(x,mfAlign,pitch,energy)
        if self.bGenerating:
            hiddenSeq,hiddenLength = self.varianceAdaptor(x)
            #hiddenSeq,hiddenLength = self.varianceAdaptor(x,mfAlign,pitch,energy)
        else:
            #hiddenSeq,hiddenLength,dLoss,pLoss,eLoss = self.varianceAdaptor(x,mfAlign,pitch,energy,lengthTensor)
            hiddenSeq,hiddenLength,d,p,e,pEmb,eEmb = self.varianceAdaptor(x,mfAlign,pitch,energy,lengthTensor)
            #hiddenSeq,hiddenLength,dLoss = self.varianceAdaptor(x,mfAlign,pitch,energy,lengthTensor)

        hiddenSeqPositionEmbeddingTensor = self.positionEmbedding(hiddenSeq,hiddenLength).to(device)#.expand(x.size()[0],-1,-1)
        hiddenSeq += hiddenSeqPositionEmbeddingTensor
        hiddenLengthTensor = self.getLengthTensor(hiddenLength).to(device)
        #hiddenSeq.masked_fill_(hiddenLengthTensor.unsqueeze(-1).expand(hiddenSeq.size()[0],hiddenSeq.size()[1],self.dDim),0)
        #print(hiddenLengthTensor,hiddenLengthTensor.size())
        hiddenSeq = self.decoderFFTBlocks(hiddenSeq,hiddenLengthTensor)
        outputSpectrogram = self.linear(hiddenSeq)
        if self.bGenerating:
            return outputSpectrogram
        

        hiddenMaskTensor = self.getLabelMaskTensor(outputSpectrogram,hiddenLengthTensor,self.outputSpectrogramDim)
        embeddingMaskTensor = self.getLabelMaskTensor(outputSpectrogram,hiddenLengthTensor,self.decoderHiddenDim)
        outputSpectrogram = outputSpectrogram.masked_select(hiddenMaskTensor)

        spectrogram = spectrogram.masked_select(hiddenMaskTensor)
        p = p.masked_select(embeddingMaskTensor)
        e = e.masked_select(embeddingMaskTensor)
        pEmb = pEmb.masked_select(embeddingMaskTensor)
        eEmb = eEmb.masked_select(embeddingMaskTensor)
        sLoss = self.loss(outputSpectrogram,spectrogram)
        dLoss = self.predictorLoss(d,mfAlign)
        pLoss = self.predictorLoss(p,pEmb)
        eLoss = self.predictorLoss(e,eEmb)
        #sLoss = 0
        #idx = 0
        #full_length = 0
        #for l in lengths:
        #    sLoss += self.loss(outputSpectrogram[idx][:l],spectrogram[idx][:l])
        #    idx+=1
        #    full_length+=l*self.outputSpectrogramDim
        #sLoss = sLoss/full_length
        #sLoss = self.loss(outputSpectrogram,spectrogram)
        loss = sLoss + dLoss + pLoss + eLoss

        return outputSpectrogram,[loss,sLoss,dLoss,pLoss,eLoss]
        #return outputSpectrogram,[loss,sLoss,dLoss,torch.zeros(1),torch.zeros(1)]

    def printWeight(self):
        print(self.encoderFFTBlocks.printWeight())

    def positionEmbedding(self,x,lengths):

        numBatch = x.size()[0]
        maxLength,_ = torch.max(lengths,0)
        #positionLength = x.size()[1]
        dDim = x.size()[-1]
        positionEmbeddingArray = np.zeros(x.size()[1:])
        #for i in range(positionLength):
        for i in range(maxLength):
            for j in range(0,dDim,2):
                positionEmbeddingArray[i][j] = math.sin(i/(10000**((2*j)/dDim)))
            for j in range(1,dDim,2):
                positionEmbeddingArray[i][j] = math.cos(i/(10000**((2*j)/dDim)))
        positionEmbeddingTensor = torch.FloatTensor(positionEmbeddingArray).repeat(numBatch,1,1)
        for i in range(numBatch):
            positionEmbeddingTensor[i][lengths[i].item():,:].fill_(0.0)
            
        return torch.FloatTensor(positionEmbeddingTensor)

    def getLengthTensor(self,lengths):   # torch.tensor([4,3,2,5,3])
        numBatch = lengths.size()[0]  
        maxLength,_ = torch.max(lengths,0)
        tmp1 = []
        for i in lengths:
            tmp2 = []
            l = i.item()
            for j in range(maxLength):
                if j < l:
                    tmp2.append(False)
                else:
                    tmp2.append(True)
            tmp1.append(tmp2)
        lengthsTensor = torch.BoolTensor(tmp1)
        #mask = mask.unsqueeze(-1).expand(numBatch,maxLength,self.dDim)
        
        return lengthsTensor

    def getLabelMaskTensor(self,a,lengths,dim):
        numBatchs,maxLengths = lengths.size()
        lengths = torch.logical_not(lengths)
        #print(lengths.size())
        #lengths = lengths.repeat(numBatchs,,self.outputSpectrogramDim)
        #print(lengths.size())
        #exit()
        mask_list = []
        for i in lengths:
            tmp = i.unsqueeze(-1).expand(maxLengths,dim)
            #mask_list.append(i.repeat(self.outputSpectrogramDim))
            mask_list.append(tmp.unsqueeze(0))
        mask = torch.cat(mask_list)

        return mask

def main():

    torch.manual_seed(1234)
    #dummyx = torch.randn(2,10,68)  # batch, time, dim
    dummyx1 = torch.randint(1,68,(1,10))
    dummyx2 = torch.randint(1,68,(1,7))
    dummyx2 = F.pad(dummyx2,(0,3),"constant")
    dummyx = torch.cat((dummyx1,dummyx2),0)
    dummyxLength = torch.tensor([10,7])
    #dummyy = torch.randn(2,20,8)
    dummyMFA1 = torch.randint(1,3,(1,10))
    dummyMFA2 = torch.randint(1,6,(1,7))
    dummyMFA2 = F.pad(dummyMFA2,(0,3),"constant")
    dummyMFA = torch.cat((dummyMFA1,dummyMFA2),0)

    #hiddenSeq = F.pad(hiddenSeq,(0,0,0,maxLengthOfHiddenSeq-seqLength),"constant")
    #print('mfa',dummyMFA,dummyMFA.size())
    #duumySpec = torch.randn(2,)
    #fastSpeech = FastSpeech2(8,8,8).eval()
    dummyPitch = torch.randint(0,256,(1,80))
    
    dummySpectrogram1 = torch.rand(1,12,128)
    dummySpectrogram1 = F.pad(dummySpectrogram1,(0,0,0,17,0,0),"constant")
    dummySpectrogram2 = torch.rand(1,29,128)
    dummySpectrogram = torch.cat((dummySpectrogram1,dummySpectrogram2),0)

    dummyPitch1 = torch.randint(1,256,(1,12))
    #dummyPitch1 = torch.rand(1,18)
    dummyPitch1 = F.pad(dummyPitch1,(0,17,0,0),"constant")
    dummyPitch2 = torch.randint(1,256,(1,29))
    #dummyPitch2 = torch.rand(1,22)
    dummyPitch = torch.cat((dummyPitch1,dummyPitch2),0)

    dummyEnergy1 = torch.randint(1,256,(1,12))
    dummyEnergy1 = F.pad(dummyEnergy1,(0,17,0,0),"constant")
    dummyEnergy2 = torch.randint(1,256,(1,29))
    dummyEnergy = torch.cat((dummyEnergy1,dummyEnergy2),0)

    fastSpeech = FastSpeech2(
            numOfEncoderBlocks=4,
            encoderFftInnerDim=1024,
            encoderKernelSize=9,
            encoderNumOfHeads=2,
            numOfDecoderBlocks=4,
            decoderFftInnerDim=1024,
            decoderKernelSize=9,
            decoderNumOfHeads=2,
            numOfPhones=69,
            phoneEmbeddingDim=256,
            encoderHiddenDim=256,
            decoderHiddenDim=256,
            outputSpectrogramDim=128,
            numOfDurationPredictorLayer=2,
            numOfPitchPredictorLayer=2,
            numOfEnergyPredictorLayer=2,
            predictorKernelSize=3,
            lossFunction='MAE',
            predictorLossFunction='MSE',
            fftDropOut=0.1,
            predictorDropOut=0.5,
            )
    print(fastSpeech)
    #output,loss,sLoss,dLoss,pLoss,eLoss = fastSpeech(dummyx,dummySpectrogram,dummyMFA,dummyxLength,dummyPitch,dummyEnergy)
    output,losss = fastSpeech(dummyx,dummySpectrogram,dummyMFA,dummyxLength,dummyPitch,dummyEnergy)
    #make_dot(losss[0],params=dict(fastSpeech.named_parameters())).render('graph2',format='png')
    #fastSpeech = FastSpeech2(bGenerating=True)
    #output = fastSpeech(dummyx,dummySpectrogram,dummyMFA,dummyxLength,dummyPitch,dummyEnergy)
    print(output)
    print('test end')

if "__main__" == __name__:
    main()


