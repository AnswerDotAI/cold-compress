#!/bin/bash

# clone the repo
git clone https://github.com/minghui-liu/cold-compress.git

# install the dependencies
cd cold-compress
pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/


git config --global credential.helper store

# huggingface login with env variable
export $(grep -v '^#' .env | xargs -d '\n')
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# setup llama3 
bash scripts/prepare_llama3.sh
# bash scripts/prepare_llama2.sh
# bash scripts/prepare_qwen2.sh

# print 'Setup done!'
echo 'Setup done!'