#/bin/bash


data_path_prefix=/workspace/mfa-kss2/
dir_list='train val test'

#prepare data
mkdir -p ./data

for i in $dir_list;do
	echo 'prepare '$i' phone and align frames'
	python3 ./utils/grid2align.py --grid_list_path=/workspace/tts/kssTTS/data/$i/${i}_textgrid.txt --save_path=/workspace/tts/kssTTS/data/$i/train_phoneAlign.pickle --data_path_prefix=$data_path_prefix
	echo 'prepare '$i' pitch, energy and mel-Spectrogram'
	python3 ./utils/wav2pitch.py --data_list_path=/workspace/tts/kssTTS/data/$i/${i}_wav.scp --data_path_prefix=$data_path_prefix --data_type=$i --save_path_prefix=/workspace/tts/kssTTS/data/$i
done

#train

#test
