import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
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
from utils import SocialDataset
from utils import *
from args import args_define
from layers import *


def extract_similar_messages(features, train_indices, new_block_indices, top_k_ratio=0.5):
    """
    DistilBERTGNN Algorithm 1, line 14: G <- extract similar messages

    At maintenance windows, select only the historical training messages that are
    most similar (cosine similarity) to the newly arrived message block.
    This discards outdated/unrelated nodes while preserving relevant knowledge.

    Args:
        features (torch.Tensor): all node feature vectors, shape (N, D)
        train_indices (torch.Tensor): indices of all current training nodes
        new_block_indices (torch.Tensor): indices of the newly arrived block nodes
        top_k_ratio (float): fraction of training nodes to keep (default 0.5)

    Returns:
        torch.Tensor: filtered training indices (most similar to new block centroid)
    """
    # Compute centroid of new block messages
    new_feats = features[new_block_indices.long()]
    centroid = F.normalize(new_feats.mean(0, keepdim=True).float(), dim=1)  # (1, D)

    # Compute cosine similarity of all training nodes vs. centroid
    all_train_feats = F.normalize(features[train_indices.long()].float(), dim=1)  # (T, D)
    sims = (all_train_feats @ centroid.T).squeeze(1)  # (T,)

    # Keep top-k most similar training nodes
    k = max(1, int(top_k_ratio * len(train_indices)))
    _, top_local_idx = sims.topk(k)
    selected_indices = train_indices[top_local_idx]

    print(f"extract_similar_messages: kept {k}/{len(train_indices)} training nodes "
          f"(top_k_ratio={top_k_ratio:.2f})")
    return selected_indices

"""
    DistilBERTGNN Model
    Paper: DistilBERT-GNN: A Powerful Approach to Social Media Event Detection
    Author: Asres Temam et. al.
    github: https://github.com/AsresTemam/DistilBERTGNN
"""

class DistilBERTGNN():
    # Inference(prediction)
    def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices=None, model=None,
            loss_fn_dgi=None, indices_to_remove=[]):
        args = args_define.args
        # make dir for graph i
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        data = SocialDataset(args.data_path, i)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension

        # Construct graph
        g = dgl.from_scipy(data.matrix)
        # graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
        num_isolated_nodes = graph_statistics(g, save_path_i)

        # if remove_obsolete is mode 1, resume or use the passed indices_to_remove to remove obsolete nodes from the graph
        if args.remove_obsolete == 1:

            if ((args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1) and (i > args.window_size)) \
                    or (indices_to_remove == [] and i > args.window_size):  # Resume indices_to_remove from the last maintain block

                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                indices_to_remove = np.load(
                    embedding_save_path + '/block_' + str(temp_i) + '/indices_to_remove.npy').tolist()

            if indices_to_remove != []:
                # remove obsolete nodes from the graph
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                # Reconstruct graph
                g = dgl.from_scipy(data.matrix)  # graph that contains tweet blocks 0, ..., i
                num_isolated_nodes = graph_statistics(g, save_path_i)

        # generate or load test mask
        if args.mask_path is None:
            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent, mask_path, len(indices_to_remove))
        else:
            test_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/test_indices.pt')

        # Suppress warning
        # g.set_n_initializer(dgl.init.zero_initializer) # Removed in DGL 2.x
        # g.readonly() # Removed in DGL 2.x

        if args.use_cuda:
            features, labels = features.cuda(), labels.cuda()
            test_indices = test_indices.cuda()
            g = g.to(features.device)

        g.ndata['features'] = features

        if (args.resume_path is not None) and (not args.resume_current) and (
                i == args.resume_point + 1):  # Resume model from the previous block and train_indices from the last initil/maintain block

            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            # Load model from resume_point
            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the previous block.")

            # Use resume_path as a flag
            args.resume_path = None

        if train_indices is None:  # train_indices is used for continue training then predict
            if args.remove_obsolete == 0 or args.remove_obsolete == 1:
                # Resume train_indices from the last initil/maintain block
                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                train_indices = torch.load(embedding_save_path + '/block_' + str(temp_i) + '/masks/train_indices.pt')
            else:
                if args.n_infer_epochs != 0:
                    print("==================================\n'continue training then predict' is unimplemented under remove_obsolete mode 2, will skip infer epochs.\n===================================\n")
                    args.n_infer_epochs = 0

        # record test nmi of all epochs
        all_test_nmi = []
        # record the time spent in seconds on direct prediction
        time_predict = []

        # Directly predict
        message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        start = time.time()
        # Infer the representations of all tweets
        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
        # Predict: conduct kMeans clustering on the test(newly inserted) nodes and report NMI
        test_nmi = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, False)
        seconds_spent = time.time() - start
        message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        all_test_nmi.append(test_nmi)
        time_predict.append(seconds_spent)
        np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Continue training then predict (leverage the features and structural info of the newly inserted block)
        if args.n_infer_epochs != 0:
            message = "\n------------ Continue training then predict on block " + str(i) + " ------------\n"
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
        # record the time spent in seconds on each batch of all infer epochs
        seconds_infer_batches = []
        # record the time spent in mins on each epoch
        mins_infer_epochs = []
        
        # Only create dataloader if we actually need infer training epochs
        if args.n_infer_epochs != 0 and train_indices is not None:
            sampler = dgl.dataloading.MultiLayerNeighborSampler([args.n_neighbors, args.n_neighbors]) # 2 hops
            # DGL DataLoader requires indices as a plain CPU torch.Tensor (with a .device attribute)
            if not isinstance(train_indices, torch.Tensor):
                train_indices_cpu = torch.tensor(train_indices, dtype=torch.long)
            else:
                train_indices_cpu = train_indices.long().cpu()
            dataloader = dgl.dataloading.DataLoader(
                g, train_indices_cpu, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0)
        else:
            dataloader = []
            
        for epoch in range(args.n_infer_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []
            for metric in metrics:
                metric.reset()
            
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time.time()
                # Load features
                batch_features = g.ndata['features'][input_nodes.cpu()].to(features.device)
                blocks = [b.to(features.device) for b in blocks]
                
                model.train()
                # forward
                if args.use_dgi:
                    pred, ret = model(blocks, batch_features) # pred: representations of the sampled nodes, ret: discriminator results
                else:
                    pred = model(blocks, batch_features)  # Representations of the sampled nodes.
                
                batch_nids = output_nodes.to(device=pred.device, dtype=torch.long)
                batch_labels = labels[batch_nids]
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                if args.use_dgi:
                    n_samples = output_nodes.size()[0]
                    lbl_1 = torch.ones(n_samples)
                    lbl_2 = torch.zeros(n_samples)
                    lbl = torch.cat((lbl_1, lbl_2), 0)
                    if args.use_cuda:
                        lbl = lbl.cuda()
                    losses_triplet.append(loss.item())
                    loss_dgi = loss_fn_dgi(ret, lbl)
                    losses_dgi.append(loss_dgi.item())
                    loss += loss_dgi
                    losses.append(loss.item())
                else:
                    losses.append(loss.item())
                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices.shape[0],
                        100. * batch_id / (train_indices.shape[0] // args.batch_size), np.mean(losses))
                    if args.use_dgi:
                        message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                        message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_infer_batches.append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_infer_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_infer_epochs.append(mins_spent)

            # Validation
            # Infer the representations of all tweets
            extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
            # Save the representations of all tweets
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            # Evaluate the model: conduct kMeans clustering on the validation and report NMI
            test_nmi = evaluate(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes, save_path_i,
                                False)
            all_test_nmi.append(test_nmi)
            # end one epoch

        # Save model (fine-tuned from the above continue training process)
        model_path = save_path_i + '/models'
        os.mkdir(model_path)
        p = model_path + '/best.pt'
        torch.save(model.state_dict(), p)
        print('Model saved.')

        # Save all test nmi
        np.save(save_path_i + '/all_test_nmi.npy', np.asarray(all_test_nmi))
        print('Saved all test nmi.')
        # Save time spent on epochs
        np.save(save_path_i + '/mins_infer_epochs.npy', np.asarray(mins_infer_epochs))
        print('Saved mins_infer_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_infer_batches.npy', np.asarray(seconds_infer_batches))
        print('Saved seconds_infer_batches.')

        return model


    # Train on initial/maintenance graphs, t == 0 or t % window_size == 0 in this paper
    def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None, loss_fn_dgi=None):
        args = args_define.args
        # make dir for graph i
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        data = SocialDataset(args.data_path, i)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension

        # Construct graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
        g = dgl.from_scipy(data.matrix) 
        num_isolated_nodes = graph_statistics(g, save_path_i)

        # if remove_obsolete is mode 1, resume or generate indices_to_remove, then remove obsolete nodes from the graph
        if args.remove_obsolete == 1:
            
            # Resume indices_to_remove from the current block
            if (args.resume_path is not None) and args.resume_current and (i == args.resume_point) and (i != 0):  
                indices_to_remove = np.load(save_path_i + '/indices_to_remove.npy').tolist()

            elif i == 0:  # generate empty indices_to_remove for initial block
                indices_to_remove = []
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            #  update graph
            else:  # generate indices_to_remove for maintenance block
                # get the indices of all training nodes
                num_all_train_nodes = np.sum(data_split[:i + 1])
                all_train_indices = np.arange(0, num_all_train_nodes).tolist()
                # get the number of old training nodes added before this maintenance
                num_old_train_nodes = np.sum(data_split[:i + 1 - args.window_size])
                # indices_to_keep: indices of nodes that are connected to the new training nodes added at this maintenance
                # (include the indices of the new training nodes)
                indices_to_keep = list(set(data.matrix.indices[data.matrix.indptr[num_old_train_nodes]:]))
                # indices_to_remove is the difference between the indices of all training nodes and indices_to_keep
                indices_to_remove = list(set(all_train_indices) - set(indices_to_keep))
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            if indices_to_remove != []:
                # remove obsolete nodes from the graph
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                # Reconstruct graph
                g = dgl.from_scipy(data.matrix)  # graph that contains tweet blocks 0, ..., i
                num_isolated_nodes = graph_statistics(g, save_path_i)

        else:

            indices_to_remove = []

        # generate or load training/validate/test masks
        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume masks from the current block

            train_indices = torch.load(save_path_i + '/masks/train_indices.pt')
            validation_indices = torch.load(save_path_i + '/masks/validation_indices.pt')
        if args.mask_path is None:

            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            train_indices, validation_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent,
                                                            mask_path, len(indices_to_remove))

        else:
            train_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/train_indices.pt')
            validation_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/validation_indices.pt')

        # Suppress warning
        # g.set_n_initializer(dgl.init.zero_initializer)
        # g.readonly()

        if args.use_cuda:
            features, labels = features.cuda(), labels.cuda()
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()
            g = g.to(features.device)

        g.ndata['features'] = features

        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume model from the current block

            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            # Load model from resume_point
            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the current block.")

            # Use resume_path as a flag
            args.resume_path = None

        elif model is None:  # Construct the initial model
            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # --- DistilBERTGNN Algorithm 1 line 14: extract similar messages at maintenance ---
        if i != 0 and args.remove_obsolete != 2:
            # Only applies to modes 0/1 where features contains accumulated historical nodes.
            # In mode 2 (Latest Message Strategy), features = current block only, so skip.
            # New block nodes are the last data_split[i] rows in the local feature matrix.
            local_n = features.shape[0]
            new_block_size = int(data_split[i])
            new_block_start_local = max(0, local_n - new_block_size)
            new_block_indices = torch.arange(new_block_start_local, local_n, dtype=torch.long)
            if args.use_cuda:
                new_block_indices = new_block_indices.cuda()
            train_indices = extract_similar_messages(
                features, train_indices, new_block_indices,
                top_k_ratio=args.top_k_ratio
            )
        # ---------------------------------------------------------------------------


        # Start training
        message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        # record the highest validation nmi ever got for early stopping
        best_vali_nmi = 1e-9
        best_epoch = 0
        wait = 0
        # record validation nmi of all epochs before early stop
        all_vali_nmi = []
        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []
        
        # Define sampler and dataloader
        sampler = dgl.dataloading.MultiLayerNeighborSampler([args.n_neighbors, args.n_neighbors]) # 2 hops
        
        for epoch in range(args.n_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []
            for metric in metrics:
                metric.reset()
            
            # Create dataloader inside loop or reset? Dataloader can be reused.
            # DGL DataLoader requires indices to be a CPU tensor
            train_indices_cpu = train_indices.cpu() if hasattr(train_indices, 'cpu') else train_indices
            dataloader = dgl.dataloading.DataLoader(
                g, train_indices_cpu, sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0)
                
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time.time()
                # Load features
                # blocks[0].srcdata['features'] = g.ndata['features'][input_nodes]
                batch_features = g.ndata['features'][input_nodes.cpu()].to(features.device)
                blocks = [b.to(features.device) for b in blocks]
                
                model.train()
                # forward
                if args.use_dgi:
                    pred, ret = model(blocks, batch_features)  # pred: representations of the sampled nodes (in the last layer of the NodeFlow), ret: discriminator results
                else:
                    pred = model(blocks, batch_features)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
                
                batch_nids = output_nodes.to(device=pred.device, dtype=torch.long)
                batch_labels = labels[batch_nids]
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                if args.use_dgi:
                    n_samples = output_nodes.size()[0]
                    lbl_1 = torch.ones(n_samples)
                    lbl_2 = torch.zeros(n_samples)
                    lbl = torch.cat((lbl_1, lbl_2), 0)
                    if args.use_cuda:
                        lbl = lbl.cuda()
                    losses_triplet.append(loss.item())
                    loss_dgi = loss_fn_dgi(ret, lbl)
                    losses_dgi.append(loss_dgi.item())
                    loss += loss_dgi
                    losses.append(loss.item())
                else:
                    losses.append(loss.item())
                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices.shape[0],
                        100. * batch_id / ((train_indices.shape[0] // args.batch_size) + 1), np.mean(losses))
                    if args.use_dgi:
                        message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                        message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)
                # end one batch

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            # Validation
            # Infer the representations of all tweets
            extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
            # Save the representations of all tweets
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            # Evaluate the model: conduct kMeans clustering on the validation and report NMI
            validation_nmi = evaluate(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                    save_path_i, True)
            all_vali_nmi.append(validation_nmi)

            # Early stop
            if validation_nmi > best_vali_nmi:
                best_vali_nmi = validation_nmi
                best_epoch = epoch
                wait = 0
                # Save model
                model_path = save_path_i + '/models'
                if (epoch == 0) and (not os.path.isdir(model_path)):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model.state_dict(), p)
                print('Best model saved after epoch ', str(epoch))
            else:
                wait += 1
            if wait == args.patience:
                print('Saved all_mins_spent')
                print('Early stopping at epoch ', str(epoch))
                print('Best model was at epoch ', str(best_epoch))
                break
            # end one epoch

        # Save all validation nmi
        np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
        # Save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        # Load the best model of the current block
        best_model_path = save_path_i + '/models/best.pt'
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded.")

        if args.remove_obsolete == 2:
            return None, indices_to_remove, model
        return train_indices, indices_to_remove, model

