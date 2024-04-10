#!/bin/bash

# Install gh CLI
type -p curl >/dev/null || sudo apt install curl -y
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y

# Authenticate with gh CLI
gh auth login

# Install nodejs
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install gitmoji, a must have
sudo npm i -g gitmoji-cli

# Download latest miniconda.
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install. -b is used to skip prompt
bash Miniconda3-latest-Linux-x86_64.sh -b

# Activate.
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# Add activation cmd to bashrc so you don't have to run the above every time.
printf '\neval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc