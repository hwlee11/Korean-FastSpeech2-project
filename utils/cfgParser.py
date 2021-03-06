

def cfgParser(path):

    f = open(path,'r')
    #contents = list()
    cfgDict = dict()
    for i in f.readlines():
        data = i.strip('\n')
        if data == '' or data[0] == '#':
            continue
        name,value=data.split('=')

        if name == 'lossFunction' or name == 'optimizer' or name == 'device':
            cfgDict[name] = value
        elif name == 'dropOut':
            cfgDict[name] = float(value)
        elif name.find('Path') != -1:
            cfgDict[name] = value
        else:
            cfgDict[name] = int(value)

    return cfgDict


def main():
    cfgParser('../cfg/FASTSPEECH2_TRAIN.cfg')

if __name__ == '__main__':
    main()
