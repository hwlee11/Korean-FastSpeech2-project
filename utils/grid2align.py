"""

Convert TextGrid file to phones frames

"""
import soundfile as sf
import math
import argparse
import pickle
import os


ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp',
           's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa',
           'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
COD = ['kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf',
           'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh',
           'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch',
           'kh', 'th', 'ph', 'h0']

phone2integer = dict()
phone_list = list()
phone_list.append('sil')
phone_list += ONS + NUC + COD
phone_list.append('ng')

for i in range(len(phone_list)):
    phone2integer[phone_list[i]] = i

SAMPLINGRATE = 44100#16000
#PAD = 2048
PAD = 1024
WINDOWSIZE = int( 30 * SAMPLINGRATE * 0.001 )
SHIFTSIZE = int( 10 * SAMPLINGRATE * 0.001 )

def find_contents_idx(contents,words):
    idx = 0
    for i in contents:
        if i.find(words) != -1:
            return idx
        idx+=1

    return -1
    

def extractAlign(file_contents,wav_path):

    start_idx=find_contents_idx(file_contents,'item [2]')
    wav,sr = sf.read(wav_path)
    wavLength = wav.shape[0]
    #finalFrameIdx = math.ceil((wavLength+PAD)/SHIFTSIZE)
    finalFrameIdx = int((wavLength)/SHIFTSIZE) + 1 

    interval_size_idx = start_idx+5#find_contents_idx(file_contents,'intervals: size')
    time = float(file_contents[start_idx+4].split(' = ')[-1])
    numOfFrames = 1 + math.floor( (time*SAMPLINGRATE - WINDOWSIZE) / SHIFTSIZE)
    #interval_size = int(file_contents[interval_size_idx].split(' = '))
    interval_size = int(file_contents[interval_size_idx].split(' = ')[-1])
    #print(numOfFrames)
    fpt = numOfFrames/time

    phones = list()
    frames = list()
    fidx = 0
    prevFrameIdx = 0
    for i in range(interval_size):
        idx = start_idx + 6 + i*4
        xmin = float(file_contents[idx+1].split(' = ')[-1])
        xmax = float(file_contents[idx+2].split(' = ')[-1])
        text = file_contents[idx+3].split(' = ')[-1]
        text = text.strip('"')
        if text == "":
            text = "sil"
        #print(idx,xmin,xmax,math.ceil(100*(xmax-xmin)),text)

        #frameLength = 1 + int(( ((xmax - xmin) * SAMPLINGRATE) - WINDOWSIZE) / SHIFTSIZE)
        #maxFrameLength = 1 + math.floor( (xmax*SAMPLINGRATE - WINDOWSIZE) / SHIFTSIZE )
        if i < interval_size-1:
            maxFrameLength = 1 + math.floor( (xmax*SAMPLINGRATE) / SHIFTSIZE )
        else:
            #maxFrameLength = math.ceil( (xmax*SAMPLINGRATE) / SHIFTSIZE )
            maxFrameLength = finalFrameIdx
            #if maxFrameLength == 376:
            #    print(xmax,maxFrameLength)
        frameLength = maxFrameLength - prevFrameIdx
        prevFrameIdx = maxFrameLength
        #frameLength = 1+ (( (xmax - xmin) * SAMPLINGRATE) // SHIFTSIZE)
        frames.append(frameLength)
        fidx += frameLength
        phones.append(phone2integer[text])
    #print(sum(frames))

    #if fidx > numOfFrames:
    #    frames[-1]-=fidx-numOfFrames

    return phones,frames

def main(args):
    file_list = open(args.grid_list_path,'r')
    wav_file_list = open(args.wav_list_path,'r')
    #directory_name = args.grid_list_path.split('/')[0]
    #print(directory_name)

    #grid_list = list()
    grid_dict = {}
    for i in file_list.readlines():
        data = i.strip('\n')
        data = os.path.join(args.data_path_prefix,data)
        #grid_list.append(data)
        file_name = data.split('/')[-1]
        file_id = file_name.split('.')[0]
        grid_dict[file_id] = data

    file_list.close()
    wav_list = list()
    for i in wav_file_list.readlines():
        data = i.strip('\n')
        data = os.path.join(args.data_path_prefix,data)
        wav_list.append(data)
    wav_file_list.close()

    data_list = list()
    #for i in grid_list:
    for i in wav_list:
        file_name = i.split('/')[-1]
        file_id = file_name.split('.')[0]
        grid_path = grid_dict[file_id]
        grid_file = open(grid_path,'r')

        file_contents = list()
        for j in grid_file.readlines():
            data = j.strip('\n')
            data = j.strip()
            file_contents.append(data)

        phones,frames = extractAlign(file_contents,i)
        data_list.append([file_id,phones,frames])
        #print(phones,frames)

    f = open(args.save_path,'wb')
    pickle.dump(data_list,f)
    f.close()
    #print(file_name,'|',' '.join(phones).strip(),'|',' '.join(frames).strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_list_path',type=str,default='../data/grid_list2.txt')
    parser.add_argument('--wav_list_path',type=str,default='./test_wav.scp')
    parser.add_argument('--data_path_prefix',type=str,default='')
    parser.add_argument('--save_path',type=str,default='./kssPhoneAlign.pickle')
    args = parser.parse_args()
    main(args)
    
