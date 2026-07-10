import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
from metrics import AverageNonzeroTripletsMetric
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics
from model import *
from utils import *
from layers import *

"""
    DistilBERTGNN Model
    Paper: DistilBERT-GNN: A Powerful Approach to Social Media Event Detection
    Author: Asres Temam et. al.
    github: https://github.com/AsresTemam/DistilBERTGNN
"""

from args import args_define

if __name__ == '__main__':
    args = args_define.args
    args.data_path = args.data_path.rstrip('/')  # normalize: remove trailing slash
    use_cuda = args.use_cuda and torch.cuda.is_available()
    print("Using CUDA:", use_cuda)

    # make dirs and save args
    if args.resume_path is None:  # build a new dir if training from scratch
        embedding_save_path = args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
        os.makedirs(embedding_save_path, exist_ok=True)

    # resume training using original dir
    else:  
        embedding_save_path = args.resume_path
    print("embedding_save_path: ", embedding_save_path)

    with open(embedding_save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load data splits
    data_split = np.load(args.data_path + '/data_split.npy')

    # Loss
    if args.use_hardest_neg:
        loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
    else:
        loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))
    if args.use_dgi:
        loss_fn_dgi = nn.BCEWithLogitsLoss()

    # Metrics
    # Counts average number of nonzero triplets found in minibatches
    metrics = [AverageNonzeroTripletsMetric()]

    # Initially, only use block 0 as training set (with explicit labels)
    train_i = 0

    # Train on initial graph
    # Resume model from the initial block or start the experiment from scratch. Otherwise (to resume from other blocks) skip this step.
    if ((args.resume_path is not None) and (args.resume_point == 0) and ( args.resume_current)) or args.resume_path is None:
        if not args.use_dgi:
            train_indices, indices_to_remove, model = DistilBERTGNN.initial_maintain(train_i, 0, data_split, metrics,\
                                                                       embedding_save_path, loss_fn)
        else:
            train_indices, indices_to_remove, model = DistilBERTGNN.initial_maintain(train_i, 0, data_split, metrics,\
                                                                       embedding_save_path, loss_fn, None, loss_fn_dgi)

    # Initialize the model, train_indices and indices_to_remove to avoid errors
    if args.resume_path is not None:
        model = None
        train_indices = None
        indices_to_remove = []

    # iterate through all blocks
    for i in range(1, data_split.shape[0]):
        # Inference (prediction)
        # Resume model from the previous, i.e., (i-1)th block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i - 1) and (not args.resume_current)) or args.resume_path is None:
            if not args.use_dgi:
                model = DistilBERTGNN.infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model, None,
                              indices_to_remove)
            else:
                model = DistilBERTGNN.infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices, model,
                              loss_fn_dgi, indices_to_remove)
        # Maintain
        # Resume model from the current, i.e., ith block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
        if ((args.resume_path is not None) and (args.resume_point == i) and (args.resume_current)) or args.resume_path is None:
            if i % args.window_size == 0:
                train_i = i
                if not args.use_dgi:
                    train_indices, indices_to_remove, model = DistilBERTGNN.initial_maintain(train_i, i, data_split, metrics,
                                                                               embedding_save_path, loss_fn, model)
                else:
                    train_indices, indices_to_remove, model = DistilBERTGNN.initial_maintain(train_i, i, data_split, metrics,
                                                                               embedding_save_path, loss_fn, model,
                                                                               loss_fn_dgi)





