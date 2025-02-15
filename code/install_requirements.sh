#!/bin/sh


conda create -n MATTER python=3.7.9 -y
conda activate MATTER
conda install -y numpy==1.20.3 pandas==1.2.4 scikit-learn=0.23.2
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt

cd /mnt/user25/Material_tokenizer/05.pretraining/MATTER

