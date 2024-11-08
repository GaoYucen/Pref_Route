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

with open('data/chengdu_data/preprocessed_test_trips_small_osmid.pkl', 'rb') as f:
    test_data = pickle.load(f)
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
        # demand_types = 0 if i < len(data)*0.8 else 1
        demand_types = 1
        new_data.append(([source, dest, trip_time, distance, demand_types], trip))
    return new_data

# 构建数据集
print('Start building dataset')
train_data = build_dataset(train_data)
test_data = build_dataset(test_data)
print('Dataset built')
# 保存数据集
with open('data/chengdu_data/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
    f.close()

with open('data/chengdu_data/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
    f.close()

# 读取数据并打印前几行
with open('data/chengdu_data/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

print(train_data[:5])