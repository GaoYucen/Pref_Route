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
        x = self.fc1(input_embed)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        return x