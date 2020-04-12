from pymongo import MongoClient
from pymongo import collection as F
from models.agent import Agent
import torch
import numpy as np
import os
import pickle
from const import MOVES, M, N, q
from copy import deepcopy, copy
from numpy.linalg import norm


def cal_corelation(state, q):
    real_state = np.round(np.cos(2 * state * np.pi/q), 15)  # + np.round(np.sin(2 * state * np.pi/q), 15)*1j
    # print(real_state)
    correlation = [np.matmul(real_state[i], np.conjugate(real_state[j])) for i in range(real_state.shape[0]) for j in range(i)]
    correlation = np.reshape(correlation, (1, np.shape(correlation)[0]))
    max_correla = norm(correlation, 1)
    return max_correla


if __name__ == '__main__':
    # min = 4
    # for i in range(100000):
    #     a = np.random.randint(0, 2, (16, 6))
    #     pending_min = cal_corelation(a, 2)
    #     if pending_min < min:
    #         min = pending_min
    #         print(min)
    #         print(a)

    # db = MongoClient().alphaSeq
    # mycollection = db['1586098437']
    # print(mycollection.count_documents({}))

    # a = np.array([[0,0,0,0,0,0],
    #               [1,0,0,1,1,1],
    #               [1,1,1,0,0,1],
    #               [0,1,1,1,1,0],
    #               [0,1,0,1,1,1],
    #               [1,1,0,0,0,0],
    #               [1,0,1,1,1,0],
    #               [0,0,1,0,0,1],
    #               [1,1,1,0,1,0],
    #               [0,1,1,1,0,1],
    #               [0,0,0,0,1,1],
    #               [1,0,0,1,0,0],
    #               [1,0,1,1,0,1],
    #               [0,0,1,0,1,0],
    #               [0,1,0,1,0,0],
    #               [1,1,0,0,1,1]])
    # b = a & np.zeros((16,6))
    # a = np.random.randint(0,4,(16,6))
    #
    # def feature_extractor(state, q):
    #     feature = []
    #     for i in range(q):
    #         feature.append((2**i & (1<<state))//2**i)
    #     return np.array(feature)
    #
    # print(a)
    # print(feature_extractor(a, 4))
    a = np.zeros((16,6),dtype=np.int64)
    print(np.conjugate(a))