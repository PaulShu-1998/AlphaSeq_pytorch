import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class ActiorCriticModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(ActiorCriticModel, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.action = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc(x))
        action_probs = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)
        return action_probs, state_values


class PolicyNetwork():
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = ActiorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def predict(self, s):
        return self.model(torch.Tensor(s))

    def update(self, returns, log_probs, state_values):
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value, Gt)
            loss += policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        action_probs, state_value = self.predict(s)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob, state_value


def actor_critic(env, estimator, n_episode, gamma=1.0):
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        total_reward_episode = defaultdict[n_episode]
        state = env.reset()
        while True:
            action, log_prob, state_value = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

            if is_done:
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma**pw * reward
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.Tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs, state_values)
                print('Episode:{}, total reward:{},time: '.format(episode, total_reward_episode[episode]))
                if total_reward_episode[episode] >= 195:
                    estimator.scheduler.step()
                break
            state = next_state