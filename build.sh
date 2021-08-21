#!/bin/bash
set -e

cd docker

if [ ! -f ./Miniconda3-py38_4.10.3-Linux-x86_64.sh ]; 
then
    echo "Miniconda file not found, downloading..."
    # wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
    chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
    echo "Miniconda file downloaded."
else
    echo "Miniconda file already present."
fi

cp ../environment.yml .
echo "Environment.yml copied."

cd ..

docker build -t real2021submission:$(date '+%Y%m%d%H%M%S') -t real2021submission:latest -f ./docker/Dockerfile2  .



