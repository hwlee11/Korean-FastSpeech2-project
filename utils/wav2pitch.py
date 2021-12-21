import pyworld as pw
import torch
import torchaudio
import soundfile as sf
import numpy as np
import math
import argparse
import pickle
import os

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def readWavFileList(listFileName,data_path_prefix):

    fileList = list()
    f = open(listFileName,'r')
    for i in f.readlines():
        data = i.strip('\n')
        data = os.path.join(data_path_prefix,data)
        fileList.append(data)
    return fileList

def linearInterpolation(f0):
    '''
    # Interpolate and normalization
    valueIdxs = np.where(f0!=0)[0]
    interpolateFunction = interp1d(valueIdxs,f0[valueIdxs],bounds_error=False,fill_value=(f0[valueIdxs[0]],f0[valueIdxs[-1]]))
    interpolatedF0 = interpolateFunction( np.arange(0,len(f0)) )
    logInterpolatedF0 = np.log(interpolatedF0)
    print(logInterpolatedF0,logInterpolatedF0.shape)
    normalizedF0 = normF0(logInterpolatedF0)
    print(normalizedF0)
    '''
    pass

def l2Norm(x):

    print(x.sum(0))
    x = x/torch.pow(x,2).sum(0)
    #mean = np.mean(x)
    #var = np.var(x)

    #x = (x-mean)/var
    return x

def muEncoding(f0,mu=255):

    sgn = np.sign(f0)
    quantaizedF0 = sgn * ( np.log(1+mu*np.absolute(f0) )/np.log(1+mu) )
    #quantaizedF0 = ((quantaizedF0+1)/2*255 + 0.5).to(int)#.astype(int)
    quantaizedF0 = ((mu+1)*quantaizedF0/torch.log(mu+torch.ones_like(quantaizedF0))).to(int)

    return quantaizedF0

def muDecoding(quantaizedF0,mu=255):

    sgn = np.sign(quantaizedF0)
    f0 = sgn*(1/mu)*( ((1+mu)**np.absolute(quantaizedF0)) -1)
    return f0

def extractF0(wave,samplingRate):

    #x,fs =sf.read(wavFile)
    f0, t = pw.dio(wave,samplingRate,frame_period = 10)
    f0 = pw.stonemask(wave,f0,t,samplingRate)
    f0 = torch.tensor(f0)
    f0 = torch.log1p(f0)
    quantaizedF0 = muEncoding(f0)
    #qf = torchaudio.functional.mu_law_encoding(f0,256)
    #f0 = muDecoding(quantaizedF0)
    return quantaizedF0

def extractEnergy(wave,samplingRate,n_fft=512,hop_length=10,win_length=30):

    win_length = int(win_length * samplingRate * 0.001)
    hop_length = int(hop_length * samplingRate * 0.001)
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))

    wave = torch.tensor(wave)
    transform = torchaudio.transforms.Spectrogram(n_fft,win_length=win_length,hop_length=hop_length)
    stft = transform(wave)

    energy = torch.sqrt(torch.pow(stft,2).sum(0))
    #energy = torch.norm(stft,dim=0)
    quantaizedEnergy = muEncoding(energy)
    print(quantaizedEnergy.size())

    return quantaizedEnergy

def extractMelSpectrogram(wave,samplingRate,n_fft=512,hop_length=10,win_length=30):

    win_length = int(win_length * samplingRate * 0.001)
    hop_length = int(hop_length * samplingRate * 0.001)
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))

    wave = torch.tensor(wave,dtype=torch.float32)
    transform = torchaudio.transforms.MelSpectrogram(samplingRate,n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    melSpectrogram = transform(wave).T
    #melSpectrogram = torchaudio.compliance.kaldi.fbank(wave,frame_length=30.,frame_shift=10.,sample_frequency=samplingRate)
    print(melSpectrogram.size())

    return melSpectrogram

def main(args):

    file_name = args.data_list_path
    save_path_prefix = args.save_path_prefix
    data_type = args.data_type
    fileList = readWavFileList(file_name,args.data_path_prefix)

    f0_dict = {}
    energy_dict = {}
    melSpec_dict = {}

    for wavFile in fileList:

        file_id = wavFile.split('/')[-1]
        file_id = file_id.split('.')[0]
        x,fs =sf.read(wavFile)
        f0 = extractF0(x,fs)
        energy = extractEnergy(x,fs)
        melSpectrogram = extractMelSpectrogram(x,fs)
        #print(melSpectrogram,melSpectrogram.size())

        #print(f0.size(),energy.size(),melSpectrogram.size())
        #if f0.size()[0] != energy.size()[0] or f0.size()[0] != melSpectrogram.size()[0]:
        #    print(wavFile)
        #    print(f0.size(),energy.size(),melSpectrogram.size())
        f0_dict[file_id] = f0
        energy_dict[file_id] = energy
        melSpec_dict[file_id] = melSpectrogram

    #f0_file_name = file_name.replace('.scp','_f0.pickle')
    f0_file_name = data_type+'_f0.pickle'
    f0_file_save_path = os.path.join(save_path_prefix,f0_file_name)
    f = open(f0_file_save_path,'wb')
    pickle.dump(f0_dict,f)
    f.close()

    #energy_file_name = file_name.replace('.scp','_energy.pickle')
    energy_file_name = data_type+'_energy.pickle'
    energy_file_save_path = os.path.join(save_path_prefix,energy_file_name)
    f = open(energy_file_save_path,'wb')
    pickle.dump(energy_dict,f)
    f.close()

    #melSpec_file_name = file_name.replace('.scp','_melSpec.pickle')
    melSpec_file_name = data_type+'_melSpec.pickle'
    melSpec_file_save_path = os.path.join(save_path_prefix,melSpec_file_name)
    f = open(melSpec_file_save_path,'wb')
    pickle.dump(melSpec_dict,f)
    f.close()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_list_path',type=str,default='test_wav.scp')
    parser.add_argument('--data_path_prefix',type=str,default='')
    parser.add_argument('--data_type',type=str,default='train')
    parser.add_argument('--save_path_prefix',type=str,default='')
    args = parser.parse_args()

    main(args)
