from pymongo import MongoClient
from models.agent import Agent
import torch
import numpy as np

if __name__ == '__main__':
    a = np.zeros((6, 5))
    b = np.zeros((6, 5))
    c = np.array([a, b])
    d = torch.from_numpy(c)
    print(d.view(-1, 1, 6, 5).shape)