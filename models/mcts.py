import numpy as np
import torch
import threading
from collections import OrderedDict
from numba import jit
from copy import deepcopy
from const import C_PUCT, EPS, ALPHA, MCTS_PARALLEL, MCTS_SIM, BATCH_SIZE_EVAL, DEVICE, M, N
from lib.CodebookEnv import DiscreteCodebook


@jit(nopython=True)
def opt_select(nodes, c_puct=C_PUCT):
    """Optimized version of the selection based of the PUCT formula"""

    total_count = 0
    for i in range(nodes.shape[0]):
        total_count += nodes[i][1]

    action_scores = np.zeros(nodes.shape[0])
    for i in range(nodes.shape[0]):
        action_scores[i] = nodes[i][0] + c_puct*nodes[i][2]*(np.sqrt(total_count)/(1+nodes[i][1]))

    equals = np.where(action_scores == np.max(action_scores))[0]
    if equals.shape[0] > 0:
        return np.random.choice(equals)
    return equals[0]


def dirichlet_noise(probs):
    dim = (probs.shape[0],)
    new_probs = (1 - EPS) * probs + EPS * np.random.dirichlet(np.full(dim, ALPHA))
    return new_probs


class Node:
    def __init__(self, parent=None, prob=None, move=None):
        """

        :param parent: parent of the node
        :param prob: probability of reaching that node, given by the policy node
        :param move: the action that leads to this node
                n: number of time this node has been visited during simulation
                w: total accumulated reward, given by the simulation
                q: mean reward (w / n)
        """
        self.prob = prob
        self.n = 0
        self.w = 0
        self.q = 0
        self.parent = parent
        self.childrens = []
        self.move = move

    def update(self, reward):
        """Update the node statistics after a playout"""
        self.w += reward
        self.q = self.w / self.n if self.n > 0 else 0

    def is_leaf(self):
        """Check whether node is a leaf or not"""
        return len(self.childrens) == 0

    def expand(self, probs):
        """Create a child node for every non-zore move probability"""
        self.childrens = [Node(parent=self, move=idx, prob=probs[idx])
                          for idx in range(probs.shape[0]) if probs[idx] > 0]


class EvaluatorThread(threading.Thread):
    def __init__(self, agent, eval_queue, result_queue, condition_search, condition_eval):
        """ Used to be able to batch evaluate positions during tree search """

        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.agent = agent
        self.condition_search = condition_search
        self.condition_eval = condition_eval

    def run(self):
        for sim in range(MCTS_SIM // MCTS_PARALLEL):

            # Wait for the eval queue to be filled by new positions to evaluate
            with self.condition_search:
                while len(self.eval_queue) < MCTS_PARALLEL:
                    self.condition_search.wait()

            with self.condition_eval:
                while len(self.result_queue) < MCTS_PARALLEL:
                    keys = list(self.eval_queue.keys())
                    max_len = BATCH_SIZE_EVAL if len(keys) > BATCH_SIZE_EVAL else len(keys)

                    # Predict the feature maps, policy and value
                    states = torch.tensor(np.array(list(self.eval_queue.values()))[0:max_len], dtype=torch.float64, device=DEVICE)
                    values, probs = self.agent.predict(states)

                    # Replace the state with the result in the eval queue and notify all the threads that the result are available
                    for idx, i in zip(keys, range(max_len)):
                        del self.eval_queue[idx]
                        self.result_queue[idx] = (probs[i].cpu().data.numpy(), values[i])

                    self.condition_eval.notifyAll()


class SearchTread(threading.Thread):
    def __init__(self, mcts, eval_queue, result_queue, thread_id, lock, condition_search, condition_eval):
        """Run a single simulation"""

        threading.Thread.__init__(self)
        self.eval_queue = eval_queue
        self.result_queue = result_queue
        self.mcts = mcts
        self.env = self._create_env()
        self.lock = lock
        self.thread_id = thread_id
        self.condition_eval = condition_eval
        self.condition_search = condition_search

    def _create_env(self):
        env = DiscreteCodebook()
        env.reset()
        return env

    def run(self):
        self.env.reset()
        current_node = self.mcts.root
        is_done = False

        # Traverse the tree until leaf
        while not current_node.is_leaf() and not is_done:
            # Select the action that maximizes the PUCT algorithm
            current_node = current_node.childrens[opt_select(np.array([[node.q, node.n, node.prob] for node in current_node.childrens]))]

            # Virtual loss since multi-threading
            with self.lock:
                current_node.n += 1

            state, _, is_done, _ = self.env.step(current_node.move)

        if not is_done:
            # Wait for the evaluator to be done
            with self.condition_eval:
                while self.thread_id not in self.result_queue.keys():
                    self.condition_eval.wait()

                # Copy the result to avoid GPU memory leak
                result = self.result_queue.pop(self.thread_id)
                probs = np.array(result[0])
                value = float(result[1])

            # #Add noise in the root node
            if not current_node.parent:
                probs = dirichlet_noise(probs)

            # #Modify probability vector depending on valid moves and normalize after that
            illegal_moves = self.env.get_illegal_moves()
            probs[illegal_moves] = 0
            total = np.sum(probs)
            probs /= total

            # Create the child nodes for the current leaf
            with self.lock:
                current_node.expand(probs)

                # Backpropagate the result of the simulation
                while current_node.parent:
                    current_node.update(value)
                    current_node = current_node.parent


class MCTS:
    def __init__(self):
        self.root = Node()
        self.tau = 1
        self.count = 1

    def _softmax(self, action_scores):
        exp = np.exp(action_scores)
        return exp / sum(exp)

    def _draw_move(self, action_scores):
        """
        Find the best move, stochastically according to some temperature constant tau
        """
        probs = self._softmax(1 / self.tau * np.log(action_scores))
        move = np.random.choice(action_scores.shape[0], p=probs)
        return move, probs

    def search(self, agent):
        """Search the best moves through the game tree with the policy and value network to update node statistics"""

        # Locking for thread synchronization
        condition_eval = threading.Condition()
        condition_search = threading.Condition()
        lock = threading.Lock()

        # Single thread for the evaluator(for now)
        eval_queue = OrderedDict()
        result_queue = {}
        evaluator = EvaluatorThread(agent, eval_queue, result_queue, condition_search, condition_eval)
        evaluator.start()

        threads = []
        # Do exactly the required number of simulation per thread
        for sim in range(MCTS_SIM // MCTS_PARALLEL):
            for idx in range(MCTS_PARALLEL):
                threads.append(SearchTread(self, eval_queue, result_queue, idx, lock, condition_search, condition_eval))
                threads[-1].start()
            for thread in threads:
                thread.join()
        evaluator.join()

        # Create the visit count vector
        action_scores = np.zeros(N)
        for node in self.root.childrens:
            action_scores[node.move] = node.n

        # Pick the best move
        final_move, final_probs = self._draw_move(action_scores)

        # Advance the root to keep the statistics of the children
        for idx in range(len(self.root.childrens)):
            if self.root.childrens[idx].move == final_move:
                break
        self.root = self.root.childrens[idx]
        self.count += 1
        if self.count > M // 3:
            self.tau = 1e-4

        return final_probs, final_move
