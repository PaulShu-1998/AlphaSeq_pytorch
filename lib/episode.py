import numpy as np
import pickle
from const import MCTS_FLAG, q, N
from models.mcts import MCTS
from .CodebookEnv import DiscreteCodebook
from .utils import _prepare_state


class Episode:
    """ A single process that is used to search by an agent """

    def __init__(self, agent, id, mcts_flag=MCTS_FLAG):
        self.id = id + 1
        self.env = self._create_env()
        self.mcts = mcts_flag
        if mcts_flag:
            self.mcts = MCTS()
        self.agent = agent

    def _create_env(self):
        env = DiscreteCodebook()
        env.reset()
        return env

    def _step(self, state):
        """ Choose a move depending on MCTS or not """
        if self.mcts:
            probs, action = self.mcts.search(self.env, self.agent)
        else:
            _, probs = self.agent.predict(state)
            probs = probs[0].cpu().data.numpy()
            action = np.random.choice([i for i in range(q**N)], p=probs)

        next_state, reward, is_done, _ = self.env.step(action)
        return next_state, reward, is_done, probs, action

    def __call__(self):
        """
        Searching by an agent and return all the states and the associated move. Also returns the reward in order to
        create the training dataset
        """

        is_done = False
        state = self.env.get_state()
        dataset = []
        cnt = 1
        while not is_done:
            # For self-play
            # state = _prepare_state(state)
            next_state, reward, is_done, probs, action = self._step(state)
            state = next_state
            dataset.append((state, probs))

        # Pickle the result because multiprocessing
        print("[EVALUATION] Episode %d done, reward %s" % (self.id, reward))
        return pickle.dumps((dataset, reward))
