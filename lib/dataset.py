import numpy as np
from torch.utils.data import Dataset, DataLoader
from const import MOVES, q, M, N
from lib.utils import feature_extractor


class SelfPlayDataset(Dataset):
    """
    Self-play dataset containing state, probabilities
    and the reward of the game.
    """

    def __init__(self):
        """ Instanciate a dataset """

        self.states = np.zeros((MOVES, M, N), dtype=np.int)
        self.probs = np.zeros((MOVES, q ** N), dtype=np.float)
        self.rewards = np.zeros(MOVES, dtype=np.float)
        self.current_len = 0

    def __len__(self):
        return self.current_len

    def __getitem__(self, idx):
        states = self.states[idx]
        probs = self.probs[idx]
        rewards = self.rewards[idx]
        # rewards = np.full((M, 1), rewards)
        return states, probs, rewards

    def update(self, game):
        """ Rotate the circular buffer to add new games at end """

        dataset = np.array(game[0])
        number_moves = dataset.shape[0]
        self.current_len = min(self.current_len + number_moves, MOVES)

        self.states = np.roll(self.states, number_moves, axis=0)
        self.states[:number_moves] = np.vstack(dataset[:, 0]).reshape([M, M, N])

        self.probs = np.roll(self.probs, number_moves, axis=0)
        self.probs[:number_moves] = np.vstack(dataset[:, 1])

        self.rewards = np.roll(self.rewards, number_moves, axis=0)
        self.rewards[:number_moves] = game[1]

        return number_moves