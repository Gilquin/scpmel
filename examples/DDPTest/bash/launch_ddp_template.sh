#!/bin/bash

# IMPORTANT:
# [path_to_examples_folder]: replace by the absolute path to DDPTest parent directory
# [path_to_anaconda3_folder]: replace by the absolute path to the anaconda3 directory

# path to script
script_path=[path_to_examples_folder]/DDPTest
# log path
log_path=$script_path/logs
# activate conda environment
source [path_to_anaconda3_folder]/etc/profile.d/conda.sh
conda activate scpmel
# launch script
launch_cmd=$ torchrun --standalone --nnodes=1 --nproc_per_node=1\
 --log_dir=$log_path -r 0:3 --tee 0:3 $script_path/ddp_example.py --use-ddp
echo"${launch_cmd}"
