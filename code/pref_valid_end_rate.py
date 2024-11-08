#%%
import geopandas as gpd
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import os
import multiprocessing as mp
from haversine import haversine

# 添加路径
import sys
sys.path.append('code')

from config import model_name
print('model_name:', model_name)

# 边信息
node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%%

# 计算边长度
def get_route_length(trip):
    total_length = 0
    for i in range(len(trip) - 1):
        start_node = trip[i]
        end_node = trip[i + 1]
        edge = edge_df[(edge_df['u'] == start_node) & (edge_df['v'] == end_node)]
        if not edge.empty:
            total_length += edge['length'].values[0]
        # else:
        #     print(f"Edge not found for nodes {start_node} to {end_node}")
    return total_length

# 读取数据集
with open('data/chengdu_data/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

# distance_max
distance_max = max([item[0][3] for item in test_data])

# 读取节点嵌入
with open('data/chengdu_data/embeddings.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

# 添加key为-1的embedding，指定dtype为float32
node_embeddings[-1] = np.array([0] * len(node_embeddings[288416374])).astype(np.float32)

# 读取node_nbrs
with open('data/chengdu_data/node_nbrs.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

# 确认node_nbrs的最大尺寸
max_nbrs = 0
for node in node_nbrs:
    if len(node_nbrs[node]) > max_nbrs:
        max_nbrs = len(node_nbrs[node])

# 将node_nbrs长度不到max_nbrs的补充到max_nbrs长度
for node in node_nbrs:
    node_nbrs[node] = list(node_nbrs[node])
    if len(node_nbrs[node]) < max_nbrs:
        node_nbrs[node] += [-1] * (max_nbrs - len(node_nbrs[node]))

# 训练
num_epoches = 10
batch_size = 64

from pref_model import Model

# # 指定cuda为device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 指定mps为device
device = torch.device('cpu')
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
# 打印当前使用的设备
print(device)
model = Model(embedding=node_embeddings).to(device)

# 加载模型参数
model.load_state_dict(torch.load('param/mac_pref_model_'+model_name+'.pth'))
model.eval()  # 设置模型为评估模式

# 准备测试数据
predicted_paths = []
original_paths = []

MAX_ITERS = 300

from collections import OrderedDict

with torch.no_grad():  # 不需要梯度计算，提高速度并减少内存消耗
    num = 0
    reach_num = 0
    for i in tqdm(range(0, len(test_data), batch_size)):
        true_paths = [item[1] for item in test_data[i:i + batch_size]]
        gens = [[t[0]] for t in true_paths]
        pending = OrderedDict({i: None for i in range(len(true_paths))})
        # for _ in tqdm(range(MAX_ITERS), desc="generating trips in lockstep", dynamic_ncols=True):
        for _ in range(MAX_ITERS):
            current_temp = [gens[i][-1] for i in pending]
            current = [c for c in current_temp for _ in node_nbrs[c]]
            pot_next = [nbr for c in current_temp for nbr in node_nbrs[c]]
            dests = [t[-1] for c, t in zip(current_temp, true_paths) for _ in (node_nbrs[c] if c in node_nbrs else [])]
            distances = [test_data[i + j][0][3] for j, c in enumerate(current_temp) for _ in node_nbrs[c]]
            demand_types = [test_data[i + j][0][4] for j, c in enumerate(current_temp) for _ in node_nbrs[c]]

            source_embed = torch.tensor(np.array([node_embeddings[node] for node in current])).to(device)
            dest_embed = torch.tensor(np.array([node_embeddings[node] for node in dests])).to(device)
            nbr_embed = torch.tensor(np.array([node_embeddings[node] for node in pot_next])).to(device)
            input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
            distances = torch.tensor(distances, dtype=torch.float32).to(device)
            demand_types = torch.tensor(demand_types, dtype=torch.float32).to(device)

            unnormalized_confidence = model(input_embed, demand_types, distances, distance_max)
            mask = torch.tensor([1 if pot_next[i] != -1 else 0 for i in range(len(pot_next))]).to(device).unsqueeze(1)
            unnormalized_confidence = unnormalized_confidence * mask
            chosen = torch.argmax(unnormalized_confidence.reshape(-1, max_nbrs), dim=1)
            chosen = chosen.detach().cpu().tolist()
            pending_trip_ids = list(pending.keys())

            for identity, choice_tmp in zip(pending_trip_ids, chosen):
                choice = node_nbrs[gens[identity][-1]][choice_tmp]
                if choice in gens[identity] or choice == -1:
                # if choice in gens[identity]:
                    del pending[identity]
                    if choice == -1:
                        num += 1
                    continue
                gens[identity].append(choice)
                if choice == true_paths[identity][-1] or haversine(node_df[node_df['osmid'] == choice][['y', 'x']].values[0], node_df[node_df['osmid'] == true_paths[identity][-1]][['y', 'x']].values[0])*1000 < 500:
                    reach_num += 1
                    del pending[identity]
                    continue

            if len(pending) == 0:
                break

        predicted_paths.extend(gens)
        original_paths.extend(true_paths)

#%% 计算抵达率
print('num:', num)
print('reach_num:', reach_num)
arrival_rate = sum([p[-1] == t[-1] for p, t in zip(predicted_paths, original_paths)]) / (len(original_paths)-num)
print("Arrival Rate:", arrival_rate)
print('reach rate:', reach_num / (len(original_paths)-num))

# 对于抵达的路径，计算precision和recall
precision_list = []
recall_list = []
for pred_path, orig_path in zip(predicted_paths, original_paths):
    if pred_path[-1] == orig_path[-1]:
        # print("Predicted Path:", pred_path)
        # print("Original Path:", orig_path)
        precision = len(set(pred_path) & set(orig_path)) / len(set(pred_path))
        recall = len(set(pred_path) & set(orig_path)) / len(set(orig_path))
        precision_list.append(precision)
        recall_list.append(recall)

precision = sum(precision_list) / len(precision_list)
recall = sum(recall_list) / len(recall_list)
print("Precision:", precision)
print("Recall:", recall)

# # 计算重合度
# pred_path_length_list = []
# for pred_path, orig_path in zip(predicted_paths, original_paths):
#     # 计算pred_path的长度
#     pred_path_length = get_route_length(pred_path)
#     pred_path_length_list.append(pred_path_length)
#
# average_pred_path_length = sum(pred_path_length_list) / len(pred_path_length_list)
# print("Average Predicted Path Length:", average_pred_path_length)