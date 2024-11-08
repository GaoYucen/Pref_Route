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
from torch.utils.data import DataLoader, Dataset

from config import model_name
# model_name = 'sim'

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

#%% 边信息
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%% 读取数据集
with open('data/chengdu_data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

#%% 计算distance_max
distance_max = max([item[0][3] for item in train_data])

#%% 读取节点嵌入
with open('data/chengdu_data/embeddings.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

#%% 添加key为-1的embedding，指定dtype为float32
node_embeddings[-1] = np.array([0] * len(node_embeddings[288416374])).astype(np.float32)

#%% 读取node_nbrs
with open('data/chengdu_data/node_nbrs.pkl', 'rb') as f:
    node_nbrs = pickle.load(f)
    f.close()

#%% 确认node_nbrs的最大尺寸
max_nbrs = 0
for node in node_nbrs:
    if len(node_nbrs[node]) > max_nbrs:
        max_nbrs = len(node_nbrs[node])

#%% 将node_nbrs长度不到max_nbrs的补充到max_nbrs长度
for node in node_nbrs:
    node_nbrs[node] = list(node_nbrs[node])
    if len(node_nbrs[node]) < max_nbrs:
        node_nbrs[node] += [-1] * (max_nbrs - len(node_nbrs[node]))

batch_size = 128

#%% 训练
from pref_model import Model

# 指定设备
# device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
device = torch.device('cpu')
# 打印当前使用的设备
print(device)
model = Model(embedding=node_embeddings).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with open('param/mac_pref_loss.txt', 'a') as f:
    #写入model_name
    f.write(f'model_name: {model_name}\n')
    f.close()

num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    random.shuffle(train_data)
    loss_list = []
    loss_ce_list = []
    loss_length_list = [1]
    for i in range(0, len(train_data), batch_size):
        optimizer.zero_grad()
        batch = [item[1] for item in train_data[i:i + batch_size]]
        source = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        distances = [train_data[i + k][0][3] for k, item in enumerate(batch) for j in range(len(item) - 1) for _ in node_nbrs[item[j]]]
        demand_types = [train_data[i + k][0][4] for k, item in enumerate(batch) for j in range(len(item) - 1) for _ in node_nbrs[item[j]]]

        source_embed = torch.tensor(np.array([node_embeddings[node] for node in source])).to(device)
        dest_embed = torch.tensor(np.array([node_embeddings[node] for node in dest])).to(device)
        nbr_embed = torch.tensor(np.array([node_embeddings[node] for node in nbr])).to(device)
        input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
        distances = torch.tensor(distances, dtype=torch.float32).to(device)
        demand_types = torch.tensor(demand_types, dtype=torch.float32).to(device)

        pred = model(input_embed, demand_types, distances, distance_max)

        # 构造mask矩阵
        mask = torch.tensor([1 if nbr[i]!= -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
        # 将pred中对应nbr == -1的部分置为0
        pred = pred * mask

        target = torch.tensor([node_nbrs[item[j]].index(item[j + 1]) for item in batch for j in range(len(item) - 1)]).to(
            device)

        # 计算原始交叉熵损失
        loss_ce = torch.nn.functional.cross_entropy(pred.view(-1, max_nbrs), target)

        # 计算更短路线损失（这里只是简单示例，假设已经有计算路线长度的函数get_route_length）
        if (demand_types == 1).any():
            predicted_routes_index = pred.view(-1, max_nbrs).argmax(dim=1)
            index_offset = 0
            predicted_routes = []
            for item in batch:
                # 使用列表推导式构建预测路线
                predicted_route = [item[0]] + [node_nbrs[item[j]][predicted_routes_index[index_offset+j]] for j in range(len(item) - 1)]
                predicted_routes.append(predicted_route)
                index_offset += len(item) - 1
            route_lengths = [get_route_length(route)/distance_max for route in predicted_routes]
            route_lengths = torch.tensor(route_lengths, dtype=torch.float32).to(device)
            # 计算预测路线长度与0的MSE Loss
            loss_length = torch.nn.functional.mse_loss(route_lengths, torch.zeros_like(route_lengths, dtype=torch.float32))
            loss_length_list.append(loss_length.item())
        else:
            loss_length = 0
            loss_length_list.append(loss_length)

        # 总损失函数，调整权重参数alpha、beta、gamma以平衡不同损失项
        loss = loss_ce + 30 * loss_length
        # loss = loss_ce
        # loss = loss_ce + math.pow(loss_length, 0.6)
        loss_list.append(loss.item())
        loss_ce_list.append(loss_ce.item())

        loss.backward()
        optimizer.step()

    print('epoch:', epoch, 'loss:', sum(loss_list)/len(loss_list), 'loss_ce:', sum(loss_ce_list)/len(loss_ce_list), 'loss_length:', sum(loss_length_list)/len(loss_length_list))
    # 存储episode和loss
    with open('param/mac_pref_loss.txt', 'a') as f:
        f.write(f'episode: {epoch}, loss: {sum(loss_list)/len(loss_list)}, loss_ce: {sum(loss_ce_list)/len(loss_ce_list)}, loss_length: {sum(loss_length_list)/len(loss_length_list)}\n')
        f.close()


# 存储模型数据
torch.save(model.state_dict(), 'param/mac_pref_model_'+model_name+'.pth')

