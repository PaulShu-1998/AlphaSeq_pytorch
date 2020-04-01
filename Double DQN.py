import torch
import numpy as np
from collections import deque, defaultdict
import random
from torch.autograd import Variable
import copy


class Estimator():
    def __init__(self, n_feat, n_state, n_action, n_hidden, lr=0.05):
        self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        self.n_feat = n_feat
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()
        for _ in range(n_action):
            model = torch.nn.Sequential(torch.nn.Linear(n_feat, n_hidden),
                                                        torch.nn.ReLU(),
                                                        torch.nn.Linear(n_hidden, 1))
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)

    def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
        torch.manual_seed(0)
        w = torch.randn((n_state, n_feat)) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * np.pi
        return w, b


class DQN():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(torch.nn.Linear(n_state, n_hidden),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(n_hidden, n_action))
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        """
        Update the weights of the DQN given a training sample
        :param s:state
        :param y: target value
        """
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))

    def target_predict(self, s):
        """
        Compute the Q values of the state for all acitons using the target network
        :param s: input state
        """
        with torch.no_grad():
            return self.model_target(torch.Tensor(s))

    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())

    def replay(self, memory, replay_size, gamma):
        """
        Experience replay with target network
        :param memory: a list of experience
        :param replay_size: the number of samples we use to update the model each time
        :param gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state.tolist()[0])
                q_values = self.predict(state).tolist()[0]
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state).detach()
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                td_targets.append(q_values)
            self.update(states, td_targets)


def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        if random.random() < epsilon:
            return random.randint(0, n_action-1)
        else:
            q_values = estimator.predict(state)
            return torch.argmax(q_values).item()
    return policy_function


def q_learning(env, estimator, n_episode, replay_size, target_update, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
    n_action = env.action_space.shape[0]
    total_reward_episode = defaultdict[n_episode]
    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        poilcy = gen_epsilon_greedy_policy(estimator, epsilon, n_action)
        state = env.reset()
        is_done = False
        while not is_done:
            action = poilcy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward

            modified_reward = next_state[0] + 0.5

            memory.append((state, action, next_state, modified_reward, is_done))

            # q_values = estimator.predict(state).tolist()
            if is_done:
                # q_values[action] = modified_reward
                # estimator.update(state, q_values)
                break
            # q_values_next = estimator.predict(next_state)
            # q_values[action] = modified_reward + gamma * torch.max(q_values_next).item()
            # estimator.update(state, q_values)
            estimator.replay(memory, replay_size, gamma)
            state = next_state
            print('Episode:{}, total reward:{},time: , epsilon: {}'.format(episode, total_reward_episode[episode], epsilon))
            epsilon = max(epsilon * epsilon_decay, 0.01)


if __name__ == '__main__':
    memory = deque(maxlen=10000)