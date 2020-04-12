import numpy as np
import pickle
from const import MCTS_FLAG, q, N
from models.mcts import MCTS
from .CodebookEnv import DiscreteCodebook, cal_corelation
from .utils import _prepare_state
import time


class Episode:
    """ A single process that is used to search by an agent """

    def __init__(self, agent, id, eval_flag=0, mcts_flag=MCTS_FLAG):
        self.id = id + 1
        self.env = self._create_env()
        self.mcts = mcts_flag
        if mcts_flag:
            self.mcts = MCTS(eval_flag)
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
        reward = -1

        while not is_done:
            # For self-play
            # state = _prepare_state(state)
            next_state, reward, is_done, probs, action = self._step(state)
            dataset.append((state, probs))
            state = next_state
        cor = cal_corelation(state)
        # Pickle the result because multiprocessing
        if reward > 0.9:
            print("################Take caution!!!#####################")
            print(state)
            print("####################################################")
            state.tofile("bestcodebook_{}.bin".format(str(int(time.time()))))
        print("[EVALUATION] Episode %d done, reward %s, cor %s" % (self.id, reward, cor))
        return pickle.dumps((dataset, reward))
