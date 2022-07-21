#!/bin/bash

MCLOUD_PATH="/home/dhoefer/python_worker_template"

export LD_LIBRARY_PATH=$MCLOUD_PATH/src/src/lib:$MCLOUD_PATH/src/linux_lib64:"$LD_LIBRARY_PATH"
export PYTHONPATH=$MCLOUD_PATH/src/src/lib:"$PYTHONPATH"

echo $LD_LIBRARY_PATH
echo $PYTHONPATH

pythonCMD="/home/dhoefer/miniconda/envs/torch/bin/python"

OMP_NUM_THREADS=8 $pythonCMD asr_worker_1.py