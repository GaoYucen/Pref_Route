import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, embedding=None):
        super(Model, self).__init__()
        self.embedding = embedding
        self.embedding_len = len(self.embedding[list(self.embedding.keys())[0]])
        self.fc0 = nn.Linear(self.embedding_len * 3 + 1, 128)
        self.fc1 = nn.Linear(self.embedding_len * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

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

        # x = self.fc1(input_embed)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        return x