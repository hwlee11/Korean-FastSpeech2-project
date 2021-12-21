import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import pickle

class KssDataSet(Dataset):

    def __init__(self,phonePath,melSpecPath,pitchPath,energyPath):

        phoneDataList = self.fileLoad(phonePath)
        self.dataIdList = list()
        self.phoneDataDict = {}
        self.mfaDataDict = {}
        for i in phoneDataList:
            self.dataIdList.append(i[0])
            self.phoneDataDict[i[0]] = torch.tensor(i[1])
            self.mfaDataDict[i[0]] = torch.tensor(i[2])

        self.melSpecDataDict = self.dictLoad(melSpecPath)
        self.pitchDataDict = self.dictLoad(pitchPath)
        self.energyDataDict = self.dictLoad(energyPath)

    def fileLoad(self,path):
        f = open(path,'rb')
        listData = pickle.load(f)
        f.close()
        return listData

    def dictLoad(self,path):
        f = open(path,'rb')
        dictData = pickle.load(f)
        f.close()
        return dictData

    def make_data(self,idx):
        dataId = self.dataIdList[idx]
        phone = self.phoneDataDict[dataId]
        melSpec = self.melSpecDataDict[dataId]
        mfa = self.mfaDataDict[dataId].type(torch.FloatTensor)
        pitch = self.pitchDataDict[dataId]
        energy = self.energyDataDict[dataId]

        return phone,melSpec,mfa,pitch,energy

    def __len__(self):
        return len(self.dataIdList)

    def __getitem__(self,idx):
        return self.make_data(idx)

def kssDataCollate(batch):
    
    batchSize = len(batch)
    lengthList = list()
    phoneList = list()
    melSpecList = list()
    mfaList = list()
    pitchList = list()
    energyList = list()
    maxMelLength = 0
    maxMfaLength = 0
    maxPitchLength = 0
    maxEnergyLength = 0
    for phone,melSpec,mfa,pitch,energy in batch:
        #print(len(pitch),sum(mfa))
        #if len(pitch) != sum(mfa).item():
        #    print(len(pitch),sum(mfa).item())
        #if len(pitch) == 529:
        #    print(len(pitch),sum(mfa).item())
        lengthList.append(phone.size()[0])
        maxMelLength = max(melSpec.size()[0],maxMelLength) # zero padding
        phoneList.append(phone)
        melSpecList.append(melSpec)
        mfaList.append(mfa)
        pitchList.append(pitch)
        energyList.append(energy)
    
    #make phone,length batch tensor
    maxLength = max(lengthList)
    lengthBatch = torch.tensor(lengthList)
    for i in range(batchSize):
        d = maxLength - phoneList[i].size()[0]
        phoneList[i] = F.pad(phoneList[i],(0,d)).unsqueeze(0)
        mfaList[i] = F.pad(mfaList[i],(0,d)).unsqueeze(0)
        d = maxMelLength - melSpecList[i].size()[0]
        melSpecList[i] = F.pad(melSpecList[i],(0,0,0,d)).unsqueeze(0)
        pitchList[i] = F.pad(pitchList[i],(0,d)).unsqueeze(0)
        energyList[i] = F.pad(energyList[i],(0,d)).unsqueeze(0)
    phoneBatch = torch.cat(phoneList)
    melSpecBatch = torch.cat(melSpecList)
    mfaBatch = torch.cat(mfaList)
    pitchBatch = torch.cat(pitchList)
    energyBatch = torch.cat(energyList)

    return phoneBatch,lengthBatch,melSpecBatch,mfaBatch,pitchBatch,energyBatch

def main():

    #data = KssDataSet('./data/val/val_phoneAlign.pickle','./data/val/val_melSpec.pickle','./data/val/val_f0.pickle','./data/val/val_energy.pickle')
    data = KssDataSet('./data/train/train_phoneAlign.pickle','./data/train/train_melSpec.pickle','./data/train/train_f0.pickle','./data/train/train_energy.pickle')
    #data = KssDataSet('./data/test/test_phoneAlign.pickle','./data/test/test_melSpec.pickle','./data/test/test_f0.pickle','./data/test/test_energy.pickle')
    #print(len(data))
    dataLoader = DataLoader(data,batch_size=1,collate_fn=kssDataCollate)
    for phone,length,melSpec,mfa,pitch,energy in dataLoader:
        pass
        #print(phone.size(),length.size(),melSpec.size(),mfa.size(),pitch.size(),energy.size())


if __name__ == "__main__":
    main()
