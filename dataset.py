import torch
from torch.utils.data import Dataset,DataLoader
import pickle

class kssDataset(Dataset):

    def __init__(self,phonePath,melSpecPath,pitchPath,energyPath):

        phoneDataList = self.fileLoad(phonePath)
        dataIdList = phoneDataList[0]
        phoneList = phoneDataList[1]
        mfaList = phoneDataList[2]
        melSpecDataDict = self.dictLoad(melSpecPath)
        pitchDataDict = self.dictLoad(pitchPath)
        energyDataDcit = self.dictLoad(energyPath)


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

    def make_data():
        pass

    def __len__(self,):
        pass
    def __getitem__(self,idx):
        pass

def main():

    data = kssDataset()


if __name__ == "__main__":
    main()
