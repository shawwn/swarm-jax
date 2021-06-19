import os
from functools import partial

#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/cuda-10.1"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["JAX_DEBUG_NANS"] = "True"

from swarm_jax.swarm_layer import NetworkPrecision

from loader import TextLoader, dtype_size
from swarm_jax.model import SwarmModel, char_layer_init, additive_weight_decay
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
p.add_argument('--dataset_length', type=int, default=0,
               help='how large is the dataset file?')
p.add_argument('--dataset_dtype', type=str, default='uint8',
               help='Does the file contain bytes (uint8), tokens (uint16), or large tokens (uint32, uint6)?')
p.add_argument('--dataset_align', type=int, default=1,
               help='When seeking to random offsets, ensure that the offsets are divisible by this.')

p.add_argument('--n_ctx', type=int, default=64,
               help='how large is the context window?')
p.add_argument('--n_head', type=int, default=8,
               help='how many attention heads?')
p.add_argument('--n_layer', type=int, default=6,
               help='how many layers? (You need to create n_layer+2 TPUs to do any training runs.)')
p.add_argument('--d_model', type=int, default=2048,
               help='hidden dim size')
p.add_argument('--vocab', type=int, default=None,
               help='vocab size. Use 256 for char GPT, 50432 for openai-style GPT, etc')

p.add_argument('--mlp_widening_factor', type=int, default=4)
p.add_argument('--attn_key_size', type=int, default=128)
p.add_argument('--attn_query_size', type=int, default=None)
p.add_argument('--attn_value_size', type=int, default=None)
p.add_argument('--attn_model_size', type=int, default=None)

p.add_argument('--lr', type=float, default=2e-4,
               help='optimizer learning rate')
p.add_argument('--beta1', type=float, default=0.9,
               help='adam optimizer beta1 momentum')
p.add_argument('--beta2', type=float, default=0.999,
               help='adam optimizer beta2 momentum')
p.add_argument('--eps', type=float, default=1e-6,
               help='adam optimizer epsilon')
p.add_argument('--clip_by_global_norm', type=float, default=1.0,
               help='clip gradients by global norm')
p.add_argument('--weight_decay', type=float, default=0.1)
p.add_argument('--batch', type=int, default=64,
               help='the global batch size')

p.add_argument('--precision_fwd', type=str, default="float32",  # default="uint16",
               help='quantize the forward pass activations before sending them over the network')
p.add_argument('--precision_rev', type=str, default="float32",  # default="uint16",
               help='quantize the reverse pass activations before sending them over the network')
p.add_argument('--precision_grad', type=str, default="float32",  # default="uint16",
               help='quantize the gradients before sending them over the network')
p.add_argument('--loss_scale', type=float, default=0,  # default=16,
               help='loss is divided by 2 ** this')

p.add_argument('--start_step', type=int, default=0,
               help='The training step to resume from')

args = p.parse_args()

if not args.vocab:
    if dtype_size(args.dataset_dtype) <= 1:
        args.vocab = 256
    else:
        args.vocab = 65536

if args.batch % 8 != 0:
    raise ValueError("--batch must be divisible by 8")

#head_info = ray.init(local_mode=True, resources={"tpu": 999})  # pretend we have infinite tpus lol
head_info = ray.init(address="auto")
pp(head_info)

batchsize = (8, args.batch // 8)
train_dataset = TextLoader(args.dataset,
                           batchsize=batchsize,
                           sample_size=args.n_ctx,
                           length=args.dataset_length,
                           dtype=args.dataset_dtype,
                           align=args.dataset_align)

chain = [
    optax.clip_by_global_norm(args.clip_by_global_norm) if args.clip_by_global_norm else None,
    optax.adam(args.lr, b1=args.beta1, b2=args.beta2, eps=args.eps),
    additive_weight_decay(args.weight_decay) if args.weight_decay else None,
]
optimizer = optax.chain(*[x for x in chain if x is not None])

prec = NetworkPrecision(fwd_act=args.precision_fwd, rev_act=args.precision_rev, grad=args.precision_grad)

model = SwarmModel(
    vocab=args.vocab,
    d_model=args.d_model,
    rev_init=partial(char_layer_init,
                     num_heads=args.n_head,
                     n_layer=args.n_layer,
                     widening_factor=args.mlp_widening_factor,
                     key_size=args.attn_key_size,
                     query_size=args.attn_query_size,
                     value_size=args.attn_value_size,
                     model_size=args.attn_model_size,
                     ),
    rev_layers=args.n_layer,
)
swarm = Swarm(model, optimizer, 2 ** args.loss_scale, train_dataset.get_samples, prec)
swarm.run(10000000, f"runs/{args.name}", f"ckpt/{args.name}", start_step=args.start_step)

ray.shutdown()
