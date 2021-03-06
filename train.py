import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import math
import argparse
import os

from FastSpeech2 import FastSpeech2
from KssDataSet import KssDataSet,kssDataCollate
from utils import cfgParser

#torch.set_printoptions(threshold=10000)

def initLossSumList(lossList):
    for i in range(len(lossList)):
        lossList[i] = 0

def learningRateScheduler(dDim,stepNum,warmupSteps):
    
    if stepNum == 0:
        return math.pow(dDim,-0.5)

    learningRate = math.pow(dDim,-0.5)*min(math.pow(stepNum,-0.5),stepNum*math.pow(warmupSteps,-1.5))
    return learningRate

def train(args):
    
    # load config
    exp_cfg = cfgParser.cfgParser(args.exp_config)
    epoch = exp_cfg['epoch']
    numOfEncodingBlocks= exp_cfg['numOfEncodingBlocks']
    encoderFftInnerDim= exp_cfg['encoderFftInnerDim']
    encoderKernelSize=exp_cfg['encoderKernelSize']
    encoderNumOfHeads=exp_cfg['encoderNumOfHeads']
    numOfDecodingBlocks=exp_cfg['numOfDecodingBlocks']
    decoderFftInnerDim=exp_cfg['decoderFftInnerDim']
    decoderKernelSize=exp_cfg['decoderKernelSize']
    decoderNumOfHeads=exp_cfg['decoderNumOfHeads']
    numOfPhones=exp_cfg['numOfPhones']
    phoneEmbeddingDim=exp_cfg['phoneEmbeddingDim']
    dDim=exp_cfg['dDim']
    outputSpectrogramDim=exp_cfg['outputSpectrogramDim']
    numOfDurationPredictorLayer=exp_cfg['numOfDurationPredictorLayer']
    numOfPitchPredictorLayer=exp_cfg['numOfPitchPredictorLayer']
    numOfEnergyPredictorLayer=exp_cfg['numOfEnergyPredictorLayer']
    predictorKernelSize=exp_cfg['predictorKernelSize']
    deviceName = exp_cfg['device']
    #optimizer
    epoch=exp_cfg['epoch']
    batchSize=exp_cfg['batchSize']
    lossFunction=exp_cfg['lossFunction']
    optimizerName=exp_cfg['optimizer']
    warmupSteps=exp_cfg['warmupSteps']
    dropOut=exp_cfg['dropOut']
    #datapath
    trainPhoneAlignPath=exp_cfg['trainPhoneAlignPath']
    trainMelSpecPath=exp_cfg['trainMelSpecPath']
    trainPitchPath=exp_cfg['trainPitchPath']
    trainEnergyPath=exp_cfg['trainEnergyPath']
    valPhoneAlignPath=exp_cfg['valPhoneAlignPath']
    valMelSpecPath=exp_cfg['valMelSpecPath']
    valPitchPath=exp_cfg['valPitchPath']
    valEnergyPath=exp_cfg['valEnergyPath']
    testPhoneAlignPath=exp_cfg['testPhoneAlignPath']
    testMelSpecPath=exp_cfg['testMelSpecPath']
    testPitchPath=exp_cfg['testPitchPath']
    testEnergyPath=exp_cfg['testEnergyPath']

    # data loader
    kssTrainDataSet = KssDataSet(trainPhoneAlignPath,trainMelSpecPath,trainPitchPath,trainEnergyPath)
    kssValDataSet = KssDataSet(valPhoneAlignPath,valMelSpecPath,valPitchPath,valEnergyPath)
    kssTestDataSet = KssDataSet(testPhoneAlignPath,testMelSpecPath,testPitchPath,testEnergyPath)
    kssTrainDataLoader = DataLoader(kssTrainDataSet,batch_size=batchSize,shuffle=True,num_workers=8,collate_fn=kssDataCollate)
    kssValDataLoader = DataLoader(kssValDataSet,batch_size=batchSize,shuffle=True,num_workers=8,collate_fn=kssDataCollate)
    kssTestDataLoader = DataLoader(kssTestDataSet,batch_size=batchSize,shuffle=True,num_workers=8,collate_fn=kssDataCollate)
    numOfTrainBatch = len(kssTrainDataLoader)
    numOfValBatch = len(kssValDataLoader)
    numOfTestBatch = len(kssTestDataLoader)
    
    # model load
    model = FastSpeech2(
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
            dDim=256,
            outputSpectrogramDim=128,
            numOfDurationPredictorLayer=2,
            numOfPitchPredictorLayer=2,
            numOfEnergyPredictorLayer=2,
            predictorKernelSize=3,
            lossFunction='MSE',
            fftDropOut=0.1,
            predictorDropOut=0.5)
    device = torch.device(deviceName)
    model.to(device)
    expName = args.exp_config.split('/')[-1].replace('.cfg','')
    savePath = os.path.join(args.save_path,expName)

    # optimizer
    stepNum = 1
    warmupSteps = warmupSteps
    learningRate = learningRateScheduler(dDim,stepNum,warmupSteps)
    #print(learningRate)
    if optimizerName == 'ADAM':
        optimizer = optim.Adam(model.parameters(),lr=learningRate,betas=(0.9,0.98),eps=1e-09)
    lossList = list()
    runningLossList = list()
    for i in range(5):
        lossList.append(0) #lossSum,mel-Loss,Duraion-Loss,Pitch-Loss,EnergyLoss
        runningLossList.append(0)

    #train loop
    for i in range(epoch):
        model.train()
        initLossSumList(lossList)
        for phone,length,melSpec,mfa,pitch,energy in kssTrainDataLoader:
            # feature 
            phone = phone.to(device)
            length = length.to(device)
            melSpec = melSpec.to(device)
            mfa = mfa.to(device)
            pitch = pitch.to(device)
            energy = energy.to(device)
            #forward
            output,losss = model(phone,melSpec,mfa,length,pitch,energy,device)
            #update
            losss[0].backward()
            lr = learningRateScheduler(dDim,stepNum,warmupSteps)
            optimizer.lr = lr
            optimizer.step()
            for j in range(len(lossList)):
                lossList[j] += losss[j].item()
                runningLossList[j] += losss[j].item()
            if stepNum % 50 == 0:
                print('epoch : %d, steps : %d, lr : %0.5f, average running loss : %0.5f, mel-loss : %0.5f, duration-loss : %0.5f, pitch-loss : %0.5f, energy-loss : %0.5f' % (i+1,stepNum,lr,runningLossList[0]/50,runningLossList[1]/50,runningLossList[2]/50,runningLossList[3]/50,runningLossList[4]/50))
                initLossSumList(runningLossList)
            stepNum+=1
        
        for j in range(5):
            lossList[j] = lossList[j]/numOfTrainBatch
        print('epoch : %d, steps : %d, lr : %0.5f, Average train loss : %0.5f, mel-loss : %0.5f, duration-loss : %0.5f, pitch-loss : %0.5f, energy-loss : %0.5f' % (i+1,stepNum,lr,lossList[0],lossList[1],lossList[2],lossList[3],lossList[4]))
        #model save
        torch.save(model.state_dict(),'%s_%d_%d.dict'%(savePath,i,stepNum))

        #validation loop
        model.eval()
        initLossSumList(lossList)
        for phone,length,melSpec,mfa,pitch,energy in kssValDataLoader:
            phone = phone.to(device)
            length = length.to(device)
            melSpec = melSpec.to(device)
            mfa = mfa.to(device)
            pitch = pitch.to(device)
            energy = energy.to(device)
            output,losss = model(phone,melSpec,mfa,length,pitch,energy,device)
            for j in range(len(lossList)):
                lossList[j] += losss[j].item()
        for j in range(5):
            lossList[j] = lossList[j]/numOfValBatch
        print('epoch : %d, steps : %d, lr : %0.5f, Average validation loss : %0.5f, mel-loss : %0.5f, duration-loss : %0.5f, pitch-loss : %0.5f, energy-loss : %0.5f' % (i+1,stepNum,lr,lossList[0],lossList[1],lossList[2],lossList[3],lossList[4]))

        initLossSumList(lossList)
        for phone,length,melSpec,mfa,pitch,energy in kssTestDataLoader:
            phone = phone.to(device)
            length = length.to(device)
            melSpec = melSpec.to(device)
            mfa = mfa.to(device)
            pitch = pitch.to(device)
            energy = energy.to(device)
            output,losss = model(phone,melSpec,mfa,length,pitch,energy,device)
            for j in range(len(lossList)):
                lossList[j] += losss[j].item()
        for j in range(5):
            lossList[j] = lossList[j]/numOfTestBatch
        print('epoch : %d, steps : %d, lr : %0.5f, Average test loss : %0.5f, mel-loss : %0.5f, duration-loss : %0.5f, pitch-loss : %0.5f, energy-loss : %0.5f' % (i+1,stepNum,lr,lossList[0],lossList[1],lossList[2],lossList[3],lossList[4]))
        
            


def main(args):
    train(args)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--exp_config',type=str,default='./cfg/FASTSPEECH2_TRAIN.cfg')    # exp config
    parse.add_argument('--save_path',type=str,default='./exp/')    # exp config
    args = parse.parse_args()
    main(args)
