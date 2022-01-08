#!/bin/bash
#PBS -N simulate
#PBS -lselect=1:ncpus=20:mem=64gb
#PBS -j oe

cd $PBS_O_WORKDIR

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ode-rnn

python simulate.py 10000 10 1 ./log/log.txt
