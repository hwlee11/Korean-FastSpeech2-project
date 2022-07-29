import torch
import torchaudio
import sys
from torchviz import make_dot
#sys.path.append('./utils/KoG2P/')

from utils.KoG2P import g2p
from utils import cfgParser
from FastSpeech2 import FastSpeech2
import matplotlib.pyplot as plt

import argparse 


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

def generator_wav(args):

    model_path = args.model_path
    rulebook_path = args.rulebook_path

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
    encoderHiddenDim=exp_cfg['encoderHiddenDim']
    decoderHiddenDim=exp_cfg['decoderHiddenDim']
    outputSpectrogramDim=exp_cfg['outputSpectrogramDim']
    numOfDurationPredictorLayer=exp_cfg['numOfDurationPredictorLayer']
    numOfPitchPredictorLayer=exp_cfg['numOfPitchPredictorLayer']
    numOfEnergyPredictorLayer=exp_cfg['numOfEnergyPredictorLayer']
    predictorKernelSize=exp_cfg['predictorKernelSize']
    #batchSize=exp_cfg['batchSize']
    lossFunction=exp_cfg['lossFunction']
    predictorLossFunction=exp_cfg['predictorLossFunction']
    fftDropOut=exp_cfg['fftDropOut']
    predictorDropOut=exp_cfg['predictorDropOut']

    deviceName = exp_cfg['device']

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
            encoderHiddenDim=256,
            decoderHiddenDim=256,
            outputSpectrogramDim=80,
            numOfDurationPredictorLayer=2,
            numOfPitchPredictorLayer=2,
            numOfEnergyPredictorLayer=2,
            predictorKernelSize=3,
            lossFunction=lossFunction,
            predictorLossFunction=predictorLossFunction,
            fftDropOut=fftDropOut,
            predictorDropOut=predictorDropOut,
            bGenerating=True)
    device = torch.device(deviceName)

    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=True )
    model.to(device)
    model.eval()

    phones =[]
    #phones = g2p.runKoG2P(' 어제 일본에서 태풍이 발생했다', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('도비는 자유에요', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('안녕하세요', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('표상은 나비다', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('빡은 탈모다', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('안녕하세요 엘론 머스크입니다', rulebook_path).split(" ")
    #phones = g2p.runKoG2P('화성 갈끄니까',rulebook_path).split(" ")
    phones = g2p.runKoG2P('기차가 예정보다 삼십 분 늦게 도착했어요', rulebook_path).split(" ")

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
    phones_ids = []
    for i in phones:
        phones_ids.append(phone2integer[i])
    phones_ids.insert(0,phone2integer['sil'])
    phones_ids.insert(15,phone2integer['sil'])
    phones_ids.append(phone2integer['sil'])
    print(phones_ids)
    phones_tensor = torch.tensor(phones_ids)
    phones_tensor = phones_tensor.to(device)
    length_tensor = torch.tensor(phones_tensor.size()[0]).to(device)
    #input_names=['hello']
    #output_names=['hello_spectorum']
    #torch.onnx.export(model, (phones_tensor.unsqueeze(0),length_tensor.unsqueeze(0)), "fastspeech2.onnx", verbose=True, input_names=input_names, output_names=output_names)
    melSpectrogram = model(phones_tensor.unsqueeze(0),lengths=length_tensor.unsqueeze(0),device=device).to('cpu').detach().transpose(1,2)#.numpy()
    temp = torch.flip(melSpectrogram,[1])
    plt.matshow(temp.squeeze(0).detach().numpy())
    plt.savefig('test_spectrum.jpg')
    plt.show()
    
    melSpectrogram = dynamic_range_decompression(melSpectrogram)
    #make_dot(melSpectrogram,params=dict(model.named_parameters())).render('graph',format='png')

    #inverMelTransform = torchaudio.transforms.InverseMelScale(n_stft=2048,sample_rate=16000)
    inverMelTransform = torchaudio.transforms.InverseMelScale(n_stft=512,n_mels=80,sample_rate=16000)
    print(melSpectrogram.size())
    spectrogram = inverMelTransform(melSpectrogram)
    print('inver mel',spectrogram.size())

    #transform = torchaudio.transforms.GriffinLim(n_fft=4094,win_length=1323,hop_length=441)
    transform = torchaudio.transforms.GriffinLim(n_fft=1022,n_iter=1024,win_length=480,hop_length=160)
    print(spectrogram)
    print(spectrogram.size())
    waveform = transform(spectrogram)
    print(waveform.size())
    sample_rate=16000
    torchaudio.save('./t436.wav',waveform,sample_rate,encoding="PCM_S",bits_per_sample=16)


def main(args):
    generator_wav(args)


if __name__ == "__main__":
    print('args')

    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path',type=str,default='',help='input save model path')
    parse.add_argument('--exp_config',type=str,default='./cfg/FASTSPEECH2_TRAIN.cfg')    # exp config
    parse.add_argument('--rulebook_path',type=str,help='')
    parse.add_argument('--input_text',type=str,default='',help='')
    args = parse.parse_args()

    main(args)

