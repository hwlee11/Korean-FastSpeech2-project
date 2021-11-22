#/bin/bash


data_path_prefix=/workspace/mfa-kss2/
config_file_path=/workspace/tts/kssTTS/cfg/FASTSPEECH2_TRAIN.cfg
model_save_path=test
dir_list='train val test'

:<<'END'
#prepare data
mkdir -p ./data

for i in $dir_list;do
	mkdir -p ./data/$i
	echo 'prepare '$i' phone and align frames'
	python3 ./utils/grid2align.py --grid_list_path=/workspace/tts/kssTTS/data/$i/${i}_textgrid.txt --save_path=/workspace/tts/kssTTS/data/$i/${i}_phoneAlign.pickle --data_path_prefix=$data_path_prefix
	echo 'prepare '$i' pitch, energy and mel-Spectrogram'
	python3 ./utils/wav2pitch.py --data_list_path=/workspace/tts/kssTTS/data/$i/${i}_wav.scp --data_path_prefix=$data_path_prefix --data_type=$i --save_path_prefix=/workspace/tts/kssTTS/data/$i
done
END
#train
mkdir -p ./exp/$model_save_path
python3 -u train.py --exp_config=$config_file_path --save_path=./exp/$model_save_path | tee temp

#generation
#python3 generation.py --model_path --str_path
