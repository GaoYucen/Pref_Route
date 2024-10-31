#%% 边信息
import geopandas as gpd

# node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%% 读取轨迹数据
import pickle
import random

with open('data/chengdu_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

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
#%%
# 构建数据集，并添加距离特征和轨迹类型标签
def build_dataset(data):
    new_data = []
    for i in range(len(data)):
        source = data[i][1][0]
        dest = data[i][1][-1]
        trip_time = data[i][2][0]
        trip = data[i][1]
        # 计算源节点到目的节点的直线距离（这里只是简单示例，实际可能需要更精确的距离计算）
        distance = get_route_length(trip)
        # demand_types# 0表示普通需求 # 1表示更短需求
        demand_types = 0 if i < len(data)*0.8 else 1
        new_data.append(([source, dest, trip_time, distance, demand_types], trip))
    return new_data

# 构建数据集
train_data = build_dataset(train_data)

#%% 计算distance_max
distance_max = max([item[0][3] for item in train_data])

#%% 读取节点嵌入
with open('data/chengdu_data/embeddings.pkl', 'rb') as f:
    node_embeddings = pickle.load(f)
    f.close()

#%% 添加key为-1的embedding，指定dtype为float32
import numpy as np
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

# #%% 读取config中定义的参数
# # 将当前目录加上/code添加到目录中
# import os
# import sys
# sys.path.append(os.getcwd() + '/code')
# import config
#
# params, _ = config.get_config()

#%% 训练
num_epoches = 10
batch_size = 64

from tqdm import tqdm
import torch
import torch.nn as nn
# from pref_model import Model
import random

class Model(nn.Module):
    def __init__(self, embedding=None):
        super(Model, self).__init__()
        self.embedding = embedding
        self.embedding_len = len(self.embedding[list(self.embedding.keys())[0]])
        self.fc0 = nn.Linear(self.embedding_len * 3 + 1, 128)
        self.fc1 = nn.Linear(self.embedding_len * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.attention = nn.MultiheadAttention(embed_dim=embedding[0].shape[0], num_heads=1)

    def forward(self, input_embed, demand_type, distances, distance_max):
        # 根据需求类型调整输入特征的处理
        shorter_mask = torch.eq(demand_type, 1)
        if shorter_mask.any():
            # 对于更短路线需求，对距离特征进行归一化处理（这里只是简单示例，可能需要更复杂的处理）
            distances = distances / distance_max
            input_embed = torch.cat((input_embed, distances.unsqueeze(-1)), dim=-1)
            x = self.fc0(input_embed)
        else:
            # 对于实走轨迹需求，这里可以添加与历史轨迹特征相关的处理（在这个示例中暂时省略具体实现）
            x = self.fc1(input_embed)
            pass
        #
        # # 使用注意力机制关注重要特征
        # input_embed, _ = self.attention(input_embed, input_embed, input_embed)

        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x

# 指定设备
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = Model(embedding=node_embeddings).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(num_epoches)):
    random.shuffle(train_data)
    loss_list = []
    loss_ce_list = []
    loss_length_list = []
    for i in range(0, len(train_data), batch_size):
        optimizer.zero_grad()
        batch = [item[1] for item in train_data[i:i + batch_size]]
        source = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        distances = [train_data[i + j // (len(item) - 1)][0][3] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        demand_types = [train_data[i + j // (len(item) - 1)][0][4] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]

        source_embed = torch.tensor([node_embeddings[node] for node in source]).to(device)
        dest_embed = torch.tensor([node_embeddings[node] for node in dest]).to(device)
        nbr_embed = torch.tensor([node_embeddings[node] for node in nbr]).to(device)
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
        else:
            loss_length = 0

        # 总损失函数，调整权重参数alpha、beta、gamma以平衡不同损失项
        loss = loss_ce + 10 * loss_length
        loss_list.append(loss.item())
        loss_ce_list.append(loss_ce.item())
        loss_length_list.append(loss_length.item())
        loss.backward()
        optimizer.step()
        # print('loss_ce:', loss_ce, ' loss_length:', loss_length, ' loss:', loss)
    print('epoch:', epoch, 'loss:', sum(loss_list)/len(loss_list), 'loss_ce:', sum(loss_ce_list)/len(loss_ce_list), 'loss_length:', sum(loss_length_list)/len(loss_length_list))
    # 存储episode和loss
    with open('param/mac_pref_loss.txt', 'a') as f:
        f.write(f'episode: {epoch}, loss: {sum(loss_list)/len(loss_list)}, loss_ce: {sum(loss_ce_list)/len(loss_ce_list)}, loss_length: {sum(loss_length_list)/len(loss_length_list)}\n')
        f.close()


# 存储模型数据
torch.save(model.state_dict(), 'param/mac_pref_model.pth')

