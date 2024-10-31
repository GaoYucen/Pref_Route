#%% 读取轨迹数据
import pickle
import random

with open('data/chengdu_data/preprocessed_train_trips_small_osmid.pkl', 'rb') as f:
    train_data = pickle.load(f)
    f.close()

# 构建数据集
for i in range(len(train_data)):
    train_data[i] = ([train_data[i][1][0], train_data[i][1][-1], train_data[i][2][0]], train_data[i][1])

#%%
# 统计一下起终点相同的traj的数量
start_end = {}

for traj in train_data:
    start = traj[0][0]
    end = traj[0][1]
    if (start, end) not in start_end:
        start_end[(start, end)] = 0
    start_end[(start, end)] += 1

#%% 输出最大最小值
print(max(start_end.values()))
print(min(start_end.values()))

#%%
import matplotlib.pyplot as plt

plt.hist(start_end.values(), bins=100)

plt.show()

