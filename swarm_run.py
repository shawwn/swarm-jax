import os

#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_DEBUG_NANS"] = "True"

from swarm_jax.swarm_layer import NetworkPrecision

from loader import TextLoader
from swarm_jax.model import SwarmCharTransformer, SwarmCharTransformerBig
from swarm_jax.swarm import Swarm

import ray
import optax

from pprint import pprint as pp

import argparse

p = argparse.ArgumentParser(
    # description=__doc__,
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
p.add_argument('--name', type=str, default='512_30L',
               help='checkpoints are saved under ckpt/{name}')
p.add_argument('--dataset', type=str, default='data/enwik9',
               help='a path to a text file to train on')
p.add_argument('--dataset-length', type=int, default=90000000,
               help='how large is the dataset file?')
p.add_argument('--n_ctx', type=int, default=64,
               help='how large is the context window?')
p.add_argument('--lr', type=float, default=2e-4,
               help='optimizer learning rate')
p.add_argument('--beta1', type=float, default=0.9,
               help='adam optimizer beta1 momentum')
p.add_argument('--beta2', type=float, default=0.99,
               help='adam optimizer beta2 momentum')
p.add_argument('--eps', type=float, default=1e-5,
               help='adam optimizer epsilon')
p.add_argument('--clip-by-global-norm', type=float, default=0.25,
               help='clip gradients by global norm')
p.add_argument('--batch', type=int, default=64,
               help='the global batch size')

args = p.parse_args()

if args.batch % 8 != 0:
    raise ValueError("--batch must be divisible by 8")

#head_info = ray.init(local_mode=True, resources={"tpu": 999})  # pretend we have infinite tpus lol
head_info = ray.init(address="auto")
pp(head_info)

batchsize = (8, args.batch // 8)
train_dataset = TextLoader(args.dataset, batchsize=batchsize, sample_size=args.n_ctx, length=args.dataset_length)

optimizer = optax.chain(
    optax.clip_by_global_norm(args.clip_by_global_norm),
    optax.adam(args.lr, b1=args.beta1, b2=args.beta2, eps=args.eps))

prec = NetworkPrecision(fwd_act="uint16", rev_act="uint16", grad="uint16")

model = SwarmCharTransformerBig
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(10000000, f"runs/{args.name}", f"ckpt/{args.name}")

ray.shutdown()
