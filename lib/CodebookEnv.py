from gym import spaces, core
import numpy as np
from numpy.linalg import norm
from const import q, M, N
from copy import copy


def cal_corelation(state):  # 未加上conjugate
    real_state = np.round(np.cos(2 * state * np.pi/q), 15)  # + np.round(np.sin(2 * state * np.pi/q), 15)*1j
    correlation = [np.matmul(real_state[i], real_state[j]) for i in range(M) for j in range(i)]
    correlation = np.reshape(correlation, (1, np.shape(correlation)[0]))
    max_correla = norm(correlation, 1)
    return max_correla


def decode_action(num, q):
    action = np.empty((N, ), dtype=np.int)
    for i in range(N):
        action[i] = num % q
        num //= q
    return action


class DiscreteCodebook(core.Env):
    """A discrete codebook environment for OpenAI gym"""
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([q for _ in range(N)])
        self.observation_space = spaces.Box(low=0, high=q-1, shape=(M, N), dtype=np.int)
        self.state = None
        self.depth = 0
        self.total_dep = M
        self.WB = N*np.sqrt((M - N) / (N * (M - 1)))
        self.uB = N
        self.a = 2.0/((self.WB - self.uB)**2)
        self.total_moves = {i for i in range(q**N)}
        self.legal_moves = {i for i in range(q**N)}
        self.encode = np.array([q ** i for i in range(N)])
        self.legal_mat = self._cre_legalmat()

    def _cre_legalmat(self):
        a = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                b = np.zeros(N, dtype=np.int)
                c = np.ones(N, dtype=np.int)
                b[i] = 1
                b[j] = 1
                c[i] = 0
                c[j] = 0
                a.append(b)
                a.append(c)
        return np.array(a)

    def step(self, action):
        obs = self._take_action(action)
        reward = self._get_reward(obs)
        done = self._get_done(action)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.state = np.zeros(self.observation_space.shape, dtype=np.int64)
        self.depth = 0
        # self.illegal_moves = None
        self.legal_moves = {i for i in range(q**N)}
        return copy(self.state)

    def _get_reward(self, state):
        if self.depth == self.total_dep:
            max_correlation = cal_corelation(state)
            reward = self.a * (max_correlation-self.uB)**2 - 1
            return reward
        return -1

    def _take_action(self, action):
        self.depth += 1
        self.state[self.depth-1, ::] = decode_action(action, q)
        # self.illegal_moves.append(action)
        # if q % 2 == 0:
        #     pre_act = np.copy(self.state[self.depth-1, ::])
        #     neg_act = (pre_act + 1) % q
        #     self.illegal_moves.append(np.sum(np.inner(neg_act, self.encode)))
        legal_moves = (np.copy(self.state[self.depth-1, ::]) + self.legal_mat) % q
        legal_moves = set(np.dot(legal_moves, self.encode))
        self.legal_moves = self.legal_moves & legal_moves

        return copy(self.state)

    def _get_done(self, action):
        if self.depth == self.total_dep:
            return True
        elif len(self.legal_moves) == 0:
            return True
        return False

    def update_ub(self, value):
        self.uB = value

    def get_illegal_moves(self):
        a = self.total_moves - self.legal_moves
        return np.array(list(a))

    def get_legal_moves(self):
        return self.legal_moves

    def get_state(self):
        return copy(self.state)

if __name__ == '__main__':
    # a = []
    # for i in range(N-1):
    #     for j in range(i+1, N):
    #         b = np.zeros(N, dtype=np.int)
    #         c = np.ones(N, dtype=np.int)
    #         b[i] = 1
    #         b[j] = 1
    #         c[i] = 0
    #         c[j] = 0
    #         a.append(b)
    #         a.append(c)
    # a = np.array(a)
    # action = decode_action(3, 2)
    # encode = np.array([q ** i for i in range(N)])
    # le = (action + a) % q
    # le = np.dot(le, encode)
    # le = set(le)
    # ille = {q**i for i in range(N)} - le
    # print(np.array(list(ille)))
    # env = DiscreteCodebook()
    # env.reset()
    # state, reward, is_done, _ = env.step(0)
    # state, reward, is_done, _ = env.step(57)
    # state, reward, is_done, _ = env.step(39)
    # state, reward, is_done, _ = env.step(30)
    # state, reward, is_done, _ = env.step(58)
    # state, reward, is_done, _ = env.step(3)
    # state, reward, is_done, _ = env.step(29)
    # state, reward, is_done, _ = env.step(36)
    # state, reward, is_done, _ = env.step(23)
    # state, reward, is_done, _ = env.step(46)
    # state, reward, is_done, _ = env.step(48)
    # state, reward, is_done, _ = env.step(9)
    # state, reward, is_done, _ = env.step(45)
    # state, reward, is_done, _ = env.step(20)
    # state, reward, is_done, _ = env.step(10)
    # state, reward, is_done, _ = env.step(51)
    # print(state)
    # print(reward)
    # print(is_done)
    # print(env.get_legal_moves())
    pass