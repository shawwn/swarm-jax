import os

from swarm_layer import NetworkPrecision

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_DEBUG_NANS"] = "True"

from loader import TextLoader
from model import SwarmCharTransformer
from swarm import Swarm

import ray
import optax

ray.init()

train_dataset = TextLoader("data/enwik8", batchsize=16, sample_size=128, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

prec = NetworkPrecision(fwd_act="uint16", rev_act="uint16", grad="uint16")

model = SwarmCharTransformer
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/uint16_2", "ckpt/uint16_2")

ray.shutdown()
