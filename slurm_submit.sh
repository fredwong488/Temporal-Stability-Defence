#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=$cyw122 # required to send email notifcations - alternatively, enter an email address
# export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
# for CephFS - remember to follow Step 2 to create folders
# /vol/gpudata/path-to-folder/myvenv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
source ~/.bashrc
# source activate
# source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime
cd /homes/cyw122/Developer/year_4/FYP/FYP-experiment-pipeline
pixi run s-ora-vr