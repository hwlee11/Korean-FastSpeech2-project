#/bin/bash


data_path_prefix=/workspace/mfa-kss2/
data_list_save_path=/workspace/kssTTS/data
config_file_path=/workspace/kssTTS/cfg/FASTSPEECH2_EXP.cfg
model_save_path=test_exp
dir_list='train val test'


#prepare data
mkdir -p ./data

for i in $dir_list;do
	mkdir -p ./data/$i
	echo 'prepare '$i' phone and align frames'
	python3 ./utils/grid2align.py --grid_list_path=/workspace/tts/kssTTS/data/$i/${i}_textgrid.txt --save_path=/workspace/tts/kssTTS/data/$i/${i}_phoneAlign.pickle --data_path_prefix=$data_path_prefix --wav_list_path=/workspace/tts/kssTTS/data/$i/${i}_wav.scp
	echo 'prepare '$i' pitch, energy and mel-Spectrogram'
	python3 ./utils/wav2pitch.py --data_list_path=/workspace/kssTTS/data/$i/${i}_wav.scp --data_path_prefix=$data_path_prefix --data_type=$i --save_path_prefix=${data_list_save_path}/${i} --fzero 0 --energy 0 --melSpec 1
done
#exit 0;

#train
mkdir -p ./exp/$model_save_path
python3 -u train.py --exp_config=$config_file_path --save_path=./exp/$model_save_path | tee temp

#generation
python3 -u generateWav.py --model_path=./exp/demo_model/FASTSPEECH2_TRAIN_DEBUG_436_421269.dict --exp_config=$config_file_path --input_text=$test_text --rulebook_path=./utils/KoG2P/rulebook.txt
#python3 -u generateWav.py --model_path=./exp/$model_save_path/your_model_path --exp_config=$config_file_path --input_text=$test_text --rulebook_path=./utils/KoG2P/rulebook.txt

