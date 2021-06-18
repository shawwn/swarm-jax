#!/usr/bin/env bash
# initializes jax and installs ray on cloud TPUs

# create tempfs for ray shared memory
sudo mkdir -p /dev/shm
sudo mount -t tmpfs -o size=100g tmpfs /dev/shm

#sudo pip install --upgrade jaxlib==0.1.59
sudo pip install --upgrade jax ray ray[default] fabric dataclasses optax git+https://github.com/deepmind/dm-haiku
git clone https://github.com/shawwn/swarm-jax ~/swarm-jax -b dev
cd ~/swarm-jax
git pull
sudo python3 setup.py develop --no-deps