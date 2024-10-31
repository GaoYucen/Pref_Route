#%% 边信息
import geopandas as gpd

# node_df = gpd.read_file('data/chengdu_data/map/nodes.shp')
edge_df = gpd.read_file('data/chengdu_data/map/edges.shp')

#%% 读取轨迹数据
import pickle
import random

with open('data/chengdu_data/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
    test_data = pickle.load(f)
    f.close()

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

test_data = build_dataset(test_data)

#%% distance_max
distance_max = max([item[0][3] for item in test_data])

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
# from model import Model
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

# # 指定cuda为device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 指定mps为device
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
model = Model(embedding=node_embeddings).to(device)

# %% 使用模型进行测试
# 加载模型参数
model.load_state_dict(torch.load('param/mac_pref_model.pth'))
model.eval()  # 设置模型为评估模式

# 准备测试数据
predictions = []
targets = []

with torch.no_grad():  # 不需要梯度计算，提高速度并减少内存消耗
    for i in range(0, len(test_data), batch_size):
        batch = [item[1] for item in test_data[i:i + batch_size]]
        source = [item[j] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        dest = [item[-1] for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        nbr = [nbr for item in batch for j in range(len(item) - 1) for nbr in node_nbrs[item[j]]]
        distances = [test_data[i + j // (len(item) - 1)][0][3] for item in batch for j in range(len(item) - 1) for nbr
                     in node_nbrs[item[j]]]
        demand_types = [test_data[i + j // (len(item) - 1)][0][4] for item in batch for j in range(len(item) - 1) for
                        nbr in node_nbrs[item[j]]]

        source_embed = torch.tensor([node_embeddings[node] for node in source]).to(device)
        dest_embed = torch.tensor([node_embeddings[node] for node in dest]).to(device)
        nbr_embed = torch.tensor([node_embeddings[node] for node in nbr]).to(device)
        input_embed = torch.cat((source_embed, dest_embed, nbr_embed), dim=1).to(device)
        distances = torch.tensor(distances, dtype=torch.float32).to(device)
        demand_types = torch.tensor(demand_types, dtype=torch.float32).to(device)

        # 进行预测
        pred = model(input_embed, demand_types, distances, distance_max)

        # 构造mask矩阵
        mask = torch.tensor([1 if nbr[i] != -1 else 0 for i in range(len(nbr))]).to(device).unsqueeze(1)
        # 将pred中对应nbr==-1的部分置为0
        pred = pred * mask

        # 获取真实目标
        true_target = torch.tensor(
            [node_nbrs[item[j]].index(item[j + 1]) for item in batch for j in range(len(item) - 1)]).to(device)

        predictions.extend(pred.view(-1, max_nbrs).argmax(dim=1).tolist())  # 保存预测结果
        targets.extend(true_target.tolist())  # 保存真实标签

# print("Predictions shape:", len(predictions))
# print("Targets shape:", len(targets))

#%% 对比predictions和targets的重叠率
overlap = sum(p == t for p, t in zip(predictions, targets))
overlap_score = overlap / len(targets)
print("Overlap Score:", overlap_score)

# %% 计算预测路径与原始路径的重合度
from collections import defaultdict

# 将预测结果转换回路径形式
predicted_paths = []
original_paths = []

# print('test data length:', len(test_data))

# 由于predictions和targets是基于展开的节点序列，我们需要将其重新组合成路径
# 遍历测试数据，根据原路径恢复预测路径和原始路径
index_offset = 0  # 用于跟踪预测结果中的索引位置
for i, test_item in enumerate(test_data):
    original_path = test_item[1]
    original_paths.append(original_path)

    # 根据预测结果构建预测路径
    predicted_path = [original_path[0]]  # 起始节点相同
    for j in range(len(original_path) - 1):
        # 寻找对应的预测结果
        next_node = node_nbrs[original_path[j]][predictions[index_offset]]
        predicted_path.append(next_node)
        index_offset += 1

    predicted_paths.append(predicted_path)

#
# print("Predicted Paths:", predicted_paths[:5])
# print("Original Paths:", original_paths[:5])
# print('len of predicted_paths:', len(predicted_paths))
# print('len of original_paths:', len(original_paths))

# 计算重合度
overlap_scores = []
pred_path_length_list = []
for pred_path, orig_path in zip(predicted_paths, original_paths):
    # 计算pred_path的长度
    pred_path_length = get_route_length(pred_path)
    pred_path_length_list.append(pred_path_length)
    overlap = sum(p == o for p, o in zip(pred_path, orig_path))
    score = overlap / len(orig_path)
    overlap_scores.append(score)

average_overlap_score = sum(overlap_scores) / len(overlap_scores)
print("Average Overlap Score:", average_overlap_score)
average_pred_path_length = sum(pred_path_length_list) / len(pred_path_length_list)
print("Average Predicted Path Length:", average_pred_path_length)