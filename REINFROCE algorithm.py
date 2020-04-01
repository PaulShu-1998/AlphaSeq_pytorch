import gym
import torch
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable

class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = nn.Sequential(nn.Linear(n_state, n_hidden),
                                   nn.ReLU(),
                                   nn.Linear(n_hidden, n_action),
                                   nn.Softmax())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        """
        Compute the action probabilities of state s using the learning model
        :param s: input state
        """
        return self.model(torch.Tensor(s))

    def update(self, advantages, log_probs):
        """
        Update the weights of the policy network given the trainning samples
        :param advantages: advantage for each step in an episode
        :param returns: return (cumulative rewards) for each step in an episode
        :param log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, advantages):
            policy_gradient.append(-log_prob * Gt)
        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
        Estimate the policy and sample an action, compute its log probability
        :param s: input state
        """
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob


class ValueNetwork():
    def __init__(self, n_state, n_hidden=50, lr=0.05):
        self.criterion = nn.MSELoss()
        self.model = nn.Sequential(nn.Linear(n_state, n_hidden),
                                   nn.ReLU(),
                                   nn.Linear(n_hidden, 1))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def update(self, s, y):
        y_pred = self.model(torch.Tensor(s))
        loss = self.criterion(y_pred, Variable(torch.Tensor(s)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, s):
        with torch.no_grad():
            return self.model(torch.Tensor(s))


def reinforce(env, estimator_policy, estimator_value, n_epispde, gamma=1.0):
    for episode in range(n_epispde):
        log_probs = []
        rewards = []
        states = []
        total_reward_episode = defaultdict[env.aciton_sapce.shape[0]]
        state = env.reset()
        while True:
            states.append(state)
            action, log_prob = estimator_policy.get_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:
                returns = []
                Gt = 0
                pw = 0
                for t in range(len(states)-1, -1, -1):
                    Gt += gamma**pw * rewards[t]
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                baseline_values = estimator_value.predict(states)
                advantages = returns - baseline_values
                # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator_value.update(states, returns)
                estimator_policy.update(advantages, log_probs)
                print('Episode:{}, total reward:{},time: '.format(episode, total_reward_episode[episode]))
                break
            state = next_state