import torch
import random
from gym import spaces, core
import numpy as np
from numpy.linalg import norm
from const import q, M, N
from copy import copy, deepcopy


def cal_corelation(state):
    real_state = np.round(np.cos(2 * state * np.pi/q), 15) + np.round(np.sin(2 * state * np.pi/q), 15)*1j
    correlation = [np.matmul(real_state[i], real_state[j]) for i in range(N) for j in range(i)]
    correlation = np.reshape(correlation, (1, np.shape(correlation)[0]))
    max_correla = norm(correlation, 1)
    return max_correla


def decode_action(num, q):
    action = np.empty((N, ), dtype=np.int64)
    for i in range(N):
        action[i] = num % q
        num //= q
    return action


class DiscreteCodebook(core.Env):
    """A discrete codebook environment for OpenAI gym"""
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([q for _ in range(N)])
        self.observation_space = spaces.Box(low=0, high=q-1, shape=(M, N), dtype=np.int64)
        self.state = None
        self.depth = 0
        self.total_dep = M
        self.WB = N*np.sqrt((M - N) / (N * (M - 1)))
        self.uB = N
        self.illegal_moves = None
        self.encode = np.array([q ** i for i in range(N)])

    def step(self, action):
        obs = self._take_action(action)
        reward = self._get_reward(obs)
        done = self._get_done()
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = np.zeros(self.observation_space.shape, dtype=np.int64)
        self.depth = 0
        self.illegal_moves = []
        return copy(self.state)

    def _get_reward(self, state):
        if self.depth == self.total_dep:
            max_correlation = cal_corelation(state)
            reward = (self.uB+self.WB-2*max_correlation)/(self.uB-self.WB)
            return reward
        return 0

    def _take_action(self, action):
        self.depth += 1
        self.state[self.depth-1, ::] = decode_action(action, q)
        self.illegal_moves.append(action)
        if q % 2 == 0:
            neg_act = np.copy(self.state[self.depth-1, ::])
            neg_act = (neg_act + 1) % q
            self.illegal_moves.append(np.sum(np.inner(neg_act, self.encode)))
        return copy(self.state)

    def _get_done(self):
        if self.depth == self.total_dep:
            return True
        return False

    def update_ub(self, value):
        self.uB = value

    def get_illegal_moves(self):
        return np.array(copy(self.illegal_moves))

    def get_state(self):
        return copy(self.state)

if __name__ == '__main__':
    env = DiscreteCodebook()
    env.reset()
    is_done = False
    while not is_done:
        action = np.random.randint(0, q**N)
        state, reward, is_done, _  = env.step(action)
        print(state, reward, is_done)