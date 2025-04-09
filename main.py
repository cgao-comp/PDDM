import sys

import args
import torch.nn as nn
import numpy as np
import torch
import random
import networkx as nx
import heapq
import pickle
import time
from tqdm import tqdm
import math
import torch.nn.functional as F
import sys
import copy

from dataReader import DataReader_snapshot
from dataLoader import GraphData
from torch.utils.data import DataLoader
from model import DiffusionSampler
from model import DiffusionModule_GAT
from GVAE import GVAEv3

from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import wasserstein_distance

def cosine_similarity(list1, list2):
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    dot_product = np.dot(vec1, vec2)

    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    similarity = dot_product / (vec1_norm * vec2_norm)

    return similarity

def get_avg_simi(graph1: nx.Graph, graph2: nx.Graph):
    num = min(len(graph1.nodes), len(graph2.nodes))
    simi_totoal = 0
    graph1_nodes_l = list(graph1.nodes)
    graph2_nodes_l = list(graph2.nodes)
    for i in range(num):
        simi_totoal += cosine_similarity(graph1.nodes[graph1_nodes_l[i]]['type_list'], graph2.nodes[graph2_nodes_l[i]]['type_list'])
    return simi_totoal / num

def graph_kernel(graph1: nx.Graph, graph2: nx.Graph, sigma: float) -> float:
    ged_generator = nx.optimize_graph_edit_distance(graph1, graph2)
    ged_puni = (next(ged_generator))

    epsilon = 1e-7
    cos_puni = 1 - get_avg_simi(graph1, graph2) + epsilon

def calculate_mmd_tqdm(graphs1: list, graphs2: list, sigma: float) -> float:
    m = len(graphs1)
    n = len(graphs2)

    total_inner_comparisons = m * m + n * n
    inner_counter = 0

    mean_within_group1 = 0
    success_time = 0
    for i in tqdm(range(m)):
        for j in range(m):
            inner_counter += 1
            mean_within_group1_part = graph_kernel(graphs1[i], graphs1[j], sigma)
            mean_within_group1 += mean_within_group1_part
            success_time += 1

    mean_within_group1 /= success_time

    success_time = 0
    mean_within_group2 = 0
    for i in tqdm(range(n)):
        for j in range(n):
            inner_counter += 1
            mean_within_group2_part = graph_kernel(graphs2[i], graphs2[j], sigma)
            mean_within_group2 += mean_within_group2_part
            success_time += 1

    mean_within_group2 /= success_time
    total_between_comparisons = m * n
    between_counter = 0
    success_time = 0
    mean_between_groups = 0
    for i in tqdm(range(m)):
        for j in range(n):
            between_counter += 1
            mean_between_groups_part = graph_kernel(graphs1[i], graphs2[j], sigma)
            mean_between_groups += mean_between_groups_part
            success_time += 1
    mean_between_groups /= success_time
    mmd = mean_within_group1 + mean_within_group2 - 2 * mean_between_groups
    return mmd

def compute_wasserstein(degrees_real, degrees_gen):
    return wasserstein_distance(degrees_real, degrees_gen)

def compute_mmd(degrees_real, degrees_gen, gamma=10, norm=0.01):    
    K_real = rbf_kernel(degrees_real.reshape(-1, 1), degrees_real.reshape(-1, 1), gamma=gamma)   
    K_gen = rbf_kernel(degrees_gen.reshape(-1, 1), degrees_gen.reshape(-1, 1), gamma=gamma)   
    K_real_gen = rbf_kernel(degrees_real.reshape(-1, 1), degrees_gen.reshape(-1, 1), gamma=gamma)  
    N_real = len(degrees_real)
    N_gen = len(degrees_gen)  
    term1 = np.sum(K_real) / (N_real * N_real)  
    term2 = np.sum(K_gen) / (N_gen * N_gen)     
    cross_term = np.sum(K_real_gen) / (N_real * N_gen)
    
    mmd = term1 + term2 - 2 * cross_term + norm * wasserstein_distance(degrees_real, degrees_gen)
    return mmd


def collate_batch(batch):  
    B = len(batch)  
        
    Chanels = batch[0][-2].shape[1]  
    N_nodes_max = batch[0][-2].shape[0]
    x = batch[0][-2]

    A = batch[0][0]
    DA = batch[0][1]
    labels = batch[0][2]  
    P = batch[0][3]

    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    influence = batch[0][5]
       
    return [DA, A, torch.where(labels==1), x, N_nodes, influence]

def pretrain_VAE(train_loader): 
    opt_vae = torch.optim.Adam([
        {'params': graphVAE.parameters(), 'lr': 0.01},
    ])
    vae_epochs = 10
    for epoch in tqdm(range(vae_epochs)):
        graphVAE.train()
        total_loss = 0
        for batch_idx, data in enumerate(train_loader):
            def weighted_binary_cross_entropy(preds, A_adj_flat, A_prop_flat):
                summ = torch.sum(A_adj_flat)
                weight_0 = summ/A_adj_flat.shape[0]
                weight_1 = 1 - weight_0
                
                weights = torch.where(
                    A_adj_flat == 1,
                    torch.full_like(A_adj_flat, weight_1),  
                    torch.full_like(A_adj_flat, weight_0)  
                )
                
                loss = nn.functional.binary_cross_entropy(preds, A_prop_flat, weight=weights)
                exp_stat = weight_0 * 1 + weight_1 * 0
                return loss, exp_stat
            opt_vae.zero_grad()
            adj_recon, mean, logvar = graphVAE(data[3][:, 0:7], data[0])
            
            A_adj = data[1]            
            adj_recon_flat = adj_recon.view(-1)
            A_adj_flat = A_adj.view(-1)           
            A_prop = data[0]
            A_prop_flat = A_prop.view(-1)
            
            recon_loss, scale = weighted_binary_cross_entropy(adj_recon_flat, A_adj_flat, A_prop_flat)
            
            kl_loss = -(1 * scale) * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + kl_loss
            loss.backward()
            opt_vae.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        

def train(train_loader, args):
    global F_score_global, exe_time
    args.device = 'cpu'
    prior_flag = False

    start = time.time()
    train_loss, n_samples = 0, 0   
    for batch_idx, data in enumerate(train_loader): 
        local_total_loss = 0.0

        T_total = data[-2].cpu()
        graph_iterative = copy.deepcopy(data[0].cpu())
        Chi_inf_sheng, user_index_sheng = torch.sort(data[-1].cpu())
        for t in range(T_total):
            opt.zero_grad()

            mean, logvar = graphVAE.encode(data[3][:, 0:7], data[0])
            z = graphVAE.reparameterize(mean, logvar)
            prior = graphVAE.decode_attn_fast(z)
            if prior_flag == True:
                prior = (prior + data[0] + data[1]) / 3
            else:
                prior = prior         
            final_miu = prior.cpu()  
            labels_t = np.zeros((T_total, 2), dtype=int)
            connections = graph_iterative[:, user_index_sheng[t]]
            for j in range(T_total):
                if connections[j] == 1:
                    labels_t[j, :] = [1, 0]
                else:
                    labels_t[j, :] = [0, 1]

            absorbing_matrix = copy.deepcopy(graph_iterative)
            graph_iterative[:, user_index_sheng[t]] = 0
            absorbing_matrix[:, user_index_sheng[t]] = 1
            absorbing_matrix[user_index_sheng[t], user_index_sheng[t]] = 0
            labels_t = torch.tensor(labels_t, dtype=float)
            weights = torch.tensor([0.9, 0.1]).to(args.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            target = torch.argmax(labels_t, dim=1)
            classfy = GAT_model(t, (prior+absorbing_matrix), data[3]).float() 
            loss = criterion(classfy, target)

            loss.backward()
            opt.step()
            local_total_loss += loss.item()
            
        local_total_loss = local_total_loss / T_total
        time_iter = time.time() - start
        train_loss += local_total_loss
        n_samples += 1

        exe_time += 1
        if batch_idx % 1 == 0 or batch_idx == len(train_loader) - 1:  
            print('Discrete graph train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), local_total_loss, train_loss / n_samples,
                time_iter / (batch_idx + 1)))

def tes_reverse_process(test_loader, args):
    args.device = 'cpu'
    pred_x_0_list = []
    start = time.time()
    train_loss, n_samples = 0, 0
    mmd_total = 0.0
    times = 0   
    for batch_idx, data in enumerate(test_loader):
        times += 1
        T_total = data[-2].cpu()

        opt.zero_grad()
        mean, logvar = graphVAE.encode(data[3][:, 0:7], torch.ones(T_total, T_total))
        z = graphVAE.reparameterize(mean, logvar)
        prior = graphVAE.decode_attn_fast(z).cpu()
        x_T = torch.randn_like(prior).cpu() + prior.cpu()

        graph_iteratives = []
        graph_iterative = torch.zeros_like(data[0])
        Chi_inf_sheng, user_index_sheng = torch.sort(data[-1].cpu(), descending=True)
        denoised_user = [int(user_index_sheng[0])]
        for t in range(1, T_total):   
            candidate = user_index_sheng[t]
            denoised_user.append(int(candidate))

            sorted_denoised_user = sorted(denoised_user)
            index_rank_in_local = sorted_denoised_user.index(candidate)
            absorbing_matrix = graph_iterative[sorted_denoised_user, :][:, sorted_denoised_user]
            absorbing_matrix[:, index_rank_in_local] = 1
            absorbing_matrix[index_rank_in_local, index_rank_in_local] = 0

            denoised_fea = data[3][sorted_denoised_user]
            classfy = GAT_model(T_total-t, (prior[sorted_denoised_user, :][:, sorted_denoised_user] + absorbing_matrix), denoised_fea).float()  

            top2_values, top2_indices = torch.topk(classfy[:, 0], 2)
            parent_of_current = sorted_denoised_user[top2_indices[0]] if index_rank_in_local != top2_indices[0] else sorted_denoised_user[top2_indices[1]]
            graph_iterative[parent_of_current][candidate] = 1
            graph_iteratives.append(graph_iterative)
        pred_x_0 = sampler(x_T, graph_iteratives, prior, data[1].cpu())  
        pred_x_0_list.append(pred_x_0)

        n_samples += 1
        exe_time += 1
    return pred_x_0_list

def test(test_loader, args):
    global F_score_global, exe_time
    args.device = 'cpu'
    start = time.time()
    train_loss, n_samples = 0, 0
    mmd_total = 0.0
    times = 0   
    for batch_idx, data in enumerate(test_loader):
        times += 1
        T_total = data[-2].cpu()

        opt.zero_grad()
        mean, logvar = graphVAE.encode(data[3][:, 0:7], torch.ones(T_total, T_total))
        z = graphVAE.reparameterize(mean, logvar)
        prior = graphVAE.decode_attn_fast(z).cpu()  

        graph_iterative = torch.zeros_like(data[0])
        Chi_inf_sheng, user_index_sheng = torch.sort(data[-1].cpu(), descending=True)
        denoised_user = [int(user_index_sheng[0])]
        for t in range(1, T_total):   
            candidate = user_index_sheng[t]
            denoised_user.append(int(candidate))

            sorted_denoised_user = sorted(denoised_user)
            index_rank_in_local = sorted_denoised_user.index(candidate)
            absorbing_matrix = graph_iterative[sorted_denoised_user, :][:, sorted_denoised_user]
            absorbing_matrix[:, index_rank_in_local] = 1
            absorbing_matrix[index_rank_in_local, index_rank_in_local] = 0

            denoised_fea = data[3][sorted_denoised_user]
            classfy = GAT_model(T_total-t, (prior[sorted_denoised_user, :][:, sorted_denoised_user] + absorbing_matrix), denoised_fea).float()  

            top2_values, top2_indices = torch.topk(classfy[:, 0], 2)
            parent_of_current = sorted_denoised_user[top2_indices[0]] if index_rank_in_local != top2_indices[0] else sorted_denoised_user[top2_indices[1]]
            graph_iterative[parent_of_current][candidate] = 1
               
        row_sums = torch.sum(graph_iterative, dim=1)        
        row_sums_list = row_sums.tolist()        
        sorted_row_sums = sorted(row_sums_list, reverse=True)
        sorted_row_sums = np.array(sorted_row_sums)
       
        row_sums_ori = torch.sum(data[0], dim=1)
        row_sums_list_ori = row_sums_ori.tolist()
        sorted_row_sums_ori = sorted(row_sums_list_ori, reverse=True)
        sorted_row_sums_ori = np.array(sorted_row_sums_ori)

        mmd = compute_mmd(sorted_row_sums_ori, sorted_row_sums)
        mmd_total += mmd
    return mmd_total/times

if __name__ == '__main__':
    with open('3a_test_100persent_not_extended.pkl', 'rb') as f:
        all_propagation_Twitter = pickle.load(f)
    all_propagation_Twitter_test = []
    for prop in all_propagation_Twitter:
        if len(prop.nodes) > 150 :
            continue
        else:
            all_propagation_Twitter_test.append(prop)
    with open('networkx_Twitter_union_graph_inf.pkl', 'rb') as f:
        Twitter_union_graph_inf = pickle.load(f)

    rnd_state = np.random.RandomState(1111)
    datareader_Twitter = DataReader_snapshot(all_propagation_Twitter_test,
                                    rnd_state=rnd_state,
                                    folds=10,
                                    union_graph_inf = Twitter_union_graph_inf)
    n_folds = 10
    for fold_id in range(n_folds):
        loaders_Twitter = []

        for split in ['train', 'test']:
            gdata_Twitter = GraphData(fold_id=fold_id,
                                      datareader=datareader_Twitter,  
                                      split=split)

            loader_Twitter = DataLoader(gdata_Twitter,  
                                        batch_size=1,  
                                        shuffle=True,  
                                        num_workers=4,
                                        collate_fn=collate_batch)  
            loaders_Twitter.append(
                loader_Twitter)                        

        print('\nDatasets: FOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders_Twitter[0].dataset),
                                                                len(loaders_Twitter[1].dataset)))

        t_start = int(1 / 10 * args.T)

        supervised = True
        unsupervised = False
        assert supervised == True and unsupervised == False, 'error!'
        
        
        if supervised == True:
            GAT_model = DiffusionModule_GAT(feature_hidden_dim=64)
            sampler = DiffusionSampler(model=GAT_model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).cpu()
            graphVAE = GVAEv3(in_channels=7, hidden_channels=32, latent_channels=16)           
            N_mask = max(list(
                len(list(all_propagation_Twitter_test[i].nodes)) for i in range(len(all_propagation_Twitter_test))))
            GLSTM_cell = GraphLSTMCell(input_dim=N_mask, hidden_dim=N_mask)
         
        else:
            forward_Trainner = None
            print('error')
            sys.exit(1)
        opt = torch.optim.Adam([
            {'params': GAT_model.parameters(), 'lr': 0.0005},
            {'params': graphVAE.parameters(), 'lr': 0.0005},
       
        ])

        opt_reverse = torch.optim.Adam([
            {'params': GLSTM_cell.parameters(), 'lr': 0.0005},
            {'params': graphVAE.parameters(), 'lr': 0.0005},
      
        ])

        all_MMD = []
        
        for epoch in range(args.epochs):
            if supervised == True:
                F_score_global_test = 0
                exe_time_test = 0
                F_score_global = 0  
                exe_time = 0  

                graphVAE.train()
                GAT_model.train()
                train(loaders_Twitter[0], args)

                if (epoch + 1) % 1 == 0:
                    g_loss = test(loaders_Twitter[1], args)
                    all_MMD.append(g_loss)
                    if len(all_MMD) >= 3:
                        print(all_MMD)
                        if all_MMD[-1] < all_MMD[-2] and all_MMD[-1] < all_MMD[-3] :
                            pred = tes_reverse_process(loaders_Twitter[1], args)
                            mmd = calculate_mmd_tqdm(loaders_Twitter[1].data['adj_list_DAG'], pred, 1.0)                       
                            print("best MMD: ", mmd)
                            break
