import os
import time
import re
import math

import mmap
import numpy as np

def dtype_size(dtype):
    bits = re.findall(r'[0-9]+$', dtype)
    if not bits:
        raise ValueError("Unknown dtype: {}".format(dtype))
    bits = int(bits[-1])
    return (bits + 7) // 8


class TextLoader():
    def __init__(self, fname, batchsize, sample_size, offset=0, length=0, dtype='uint8'):
        self.f = open(fname, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), length=length, offset=offset)
        self.file_size = os.stat(fname).st_size
        if length > 0:
            self.file_size = min(self.file_size, length)
        self.bs = np.product(batchsize)

        if isinstance(batchsize, tuple):
            self.batch_shape = batchsize
        else:
            self.batch_shape = (batchsize,)
        self.ss = sample_size

        self.dtype = dtype
        self.np_mm = np.memmap(fname, dtype=self.dtype, mode='r', shape=(self.file_total,))

    @property
    def file_total(self):
        """How many samples are in the file?"""
        return self.file_size // dtype_size(self.dtype)

    def get_samples(self):
        sample = np.random.randint(0, self.file_total - 2 - self.ss, self.bs)
        batch = np.zeros((self.bs, self.ss + 1))

        for i in range(self.ss + 1):
            batch[:, i] = self.np_mm[sample + i]

        target = batch[:, 1:].astype(np.uint32)
        target = target.reshape(self.batch_shape + (self.ss,))

        obs = batch[:, :-1].astype(np.uint32)
        obs = obs.reshape(self.batch_shape + (self.ss,))

        return {"target": target, "obs": obs}


if __name__ == "__main__":
    tl = TextLoader("data/enwik9", batchsize=(8, 128), sample_size=128)
    np.sum(tl.np_mm)
    print("preload done")

    for i in range(100):
        tl.get_samples()

    print("warmup done")

    start = time.time()

    it = 1000

    for i in range(it):
        tl.get_samples()

    t = time.time() - start
    print(f"samples done in {t} s")
    print(f"{tl.bs * it/t} eg/s")

