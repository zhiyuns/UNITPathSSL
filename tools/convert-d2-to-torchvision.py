#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import pickle as pkl
import sys
import torch
import pickle
import collections

"""
Usage:
  # download one of the ResNet{18,34,50,101,152} models from torchvision:
  wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O r50.pth
  # run the conversion
  ./convert-torchvision-to-d2.py r50.pth r50.pkl

  # Then, use r50.pkl with the following changes in config:

MODEL:
  WEIGHTS: "/path/to/r50.pkl"
  PIXEL_MEAN: [123.675, 116.280, 103.530]
  PIXEL_STD: [58.395, 57.120, 57.375]
  RESNETS:
    DEPTH: 50
    STRIDE_IN_1X1: False
INPUT:
  FORMAT: "RGB"

  These models typically produce slightly worse results than the
  pre-trained ResNets we use in official configs, which are the
  original ResNet models released by MSRA.
"""

if __name__ == "__main__":
    input = sys.argv[1]

    with open(input, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    obj = obj['model']
    newmodel = collections.OrderedDict()
    for k in list(obj.keys()):
        if 'num_batches_tracked' in k:
            continue
        old_k = k
        k = k.replace('backbone.bottom_up.', '')
        k = k.replace('stem.', '')
        for t in [1, 2, 3, 4]:
            k = k.replace("res{}".format(t + 1), "layer{}".format(t))
        for t in [1, 2, 3]:
            k = k.replace("conv{}.norm".format(t), "bn{}".format(t))
        k = k.replace("shortcut.norm", "downsample.1")
        k = k.replace("shortcut", "downsample.0")
        print(old_k, "->", k)
        newmodel[k] = torch.Tensor(obj.pop(old_k).cpu())

    torch.save(newmodel, sys.argv[2] ,_use_new_zipfile_serialization=False)

