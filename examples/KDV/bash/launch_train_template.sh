#!/bin/bash
# [path_to_examples_folder]: replace by the absolute path to KDV parent directory
# [path_to_anaconda3_folder]: replace by the absolute path to the anaconda3 directory

export WORLD_SIZE='1'

# KDV folder path
folder_path=[path_to_examples_folder]/KDV
echo $folder_path
# data path
data_path=$folder_path/data/KDV_LES_1600.0.hdf5
# checkpoint_dir
checkpoint_path=$folder_path/checkpoints/classic
echo $checkpoint_path
# save directory
save_path=$folder_path/data/gradients/classic
echo $save_path
# activate conda environment
source [path_to_anaconda3_folder]/etc/profile.d/conda.sh
conda activate scpmel
# launch script
launch_cmd=$python $folder_path/main_train.py --run 0 --epochs 3\
 --start-epoch 0 --batch-size 64 --save_dir $save_path\
 --checkpoint_dir $checkpoint_path --seed 1990 --gpu 0\
 $data_path
echo"${launch_cmd}"
