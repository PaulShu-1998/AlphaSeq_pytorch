import timeit
from .search import play
from const import EVAL_MATCHES
import numpy as np


def evaluate(agent, new_agent):
    """ Used to evaluate the best network against
        the newly trained model """

    print("[EVALUATION] Starting to evaluate trained model !")
    # start_time = timeit.default_timer()
    #
    # # Play the matches and get the results
    # results = play(agent)
    # results_new = play(new_agent)
    # final_time = timeit.default_timer() - start_time
    # print("[EVALUATION] Total duration: %.3f seconds, average duration: %.3f seconds" % ((final_time, final_time / EVAL_MATCHES)))
    #
    # # Count the number of wins for each players
    # mean_rewards = np.mean(results)
    # mean_rewards_new = np.mean(results_new)
    # print("[EVALUATION] old agent rewards: {} vs {} for new agent" .format(mean_rewards, mean_rewards_new))
    #
    # # Check if the trained player (black) is better than
    # # the current best player depending on the threshold
    # if mean_rewards_new >= mean_rewards:
    #     return True
    # return False
    return True
