"""

Convert TextGrid file to phones frames

"""
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
phone_list.append('sil')
phone_list = ONS + NUC + COD
phone_list.append('ng')

for i in range(len(phone_list)):
    phone2integer[phone_list[i]] = i

SAMPLINGRATE = 16000
WINDOWSIZE = int( 30 * SAMPLINGRATE * 0.001 )
SHIFTSIZE = int( 10 * SAMPLINGRATE * 0.001 )

def find_contents_idx(contents,words):
    idx = 0
    for i in contents:
        if i.find(words) != -1:
            return idx
        idx+=1

    return -1
    

def extractAlign(file_contents):

    start_idx=find_contents_idx(file_contents,'item [2]')

    interval_size_idx = start_idx+5#find_contents_idx(file_contents,'intervals: size')
    time = float(file_contents[start_idx+4].split(' = ')[-1])
    numOfFrames =1 + math.floor( (time*SAMPLINGRATE - WINDOWSIZE) / SHIFTSIZE)
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
        maxFrameLength = 1 + math.floor( (xmax*SAMPLINGRATE) / SHIFTSIZE )
        frameLength = maxFrameLength - prevFrameIdx
        prevFrameIdx = maxFrameLength
        #frameLength = 1+ (( (xmax - xmin) * SAMPLINGRATE) // SHIFTSIZE)
        frames.append(frameLength)
        fidx += frameLength
        phones.append(phone2integer[text])

    #if fidx > numOfFrames:
    #    frames[-1]-=fidx-numOfFrames

    return phones,frames

def main(args):
    file_list = open(args.grid_list_path,'r')
    #directory_name = args.grid_list_path.split('/')[0]
    #print(directory_name)

    grid_list = list()
    for i in file_list.readlines():
        data = i.strip('\n')
        data = os.path.join(args.data_path_prefix,data)
        grid_list.append(data)
        

    data_list = list()
    for i in grid_list:
        grid_file = open(i,'r')
        file_name = i.split('/')[-1]
        file_id = file_name.split('.')[0]

        file_contents = list()
        for j in grid_file.readlines():
            data = j.strip('\n')
            data = j.strip()
            file_contents.append(data)

        phones,frames = extractAlign(file_contents)
        data_list.append([file_id,phones,frames])
        #print(phones,frames)

    f = open(args.save_path,'wb')
    pickle.dump(data_list,f)
    f.close()
    #print(file_name,'|',' '.join(phones).strip(),'|',' '.join(frames).strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_list_path',type=str,default='grid_list.txt')
    parser.add_argument('--data_path_prefix',type=str,default='')
    parser.add_argument('--save_path',type=str,default='./kssPhoneAlign.pickle')
    args = parser.parse_args()
    main(args)
    
