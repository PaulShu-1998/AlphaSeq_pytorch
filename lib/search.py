import pickle
import time
import timeit
from copy import deepcopy
from const import PARALLEL_EVAL, PARALLEL_SELF_PLAY, SELF_PLAY_MATCH, EVAL_MATCHES
from pymongo import MongoClient
from .utils import get_agent, load_agent, _prepare_state
from .process import create_episodes


def self_play(current_time, loaded_version):
    """
    Used to create a learning dataset for the value and policy network.
    Play against itself and backtrack the winner to maximize winner moves
    probabilities
    """

    # Init database connection
    client = MongoClient()
    collection = client.alphaSeq[current_time]

    game_id = 0
    current_version = 1
    agent = False

    while True:

        # Load the agent when restarting training
        if loaded_version:
            agent, checkpoint = load_agent(current_time, loaded_version)
            game_id = collection.find().count()
            current_version = checkpoint['version'] + 1
            loaded_version = False
        else:
            agent, checkpoint = get_agent(current_time, current_version)
            if agent:
                current_version = checkpoint['version'] + 1

        # Waiting for the first player to be saved
        print("[PLAY] Current improvement level: %d" % current_version)
        if current_version == 1 and not agent:
            print("[PLAY] Waiting for first agent")
            time.sleep(5)
            continue

        # Create the self-play match queue of processes
        queue, results = create_episodes(agent, cores=PARALLEL_SELF_PLAY, match_number=SELF_PLAY_MATCH)
        print("[PLAY] Starting to fetch fresh episodes")
        start_time = timeit.default_timer()

        try:
            queue.join()

            # Collect the results and push them in the database
            for _ in range(SELF_PLAY_MATCH):
                result = results.get()
                if result:
                    collection.insert({
                        "game": result,
                        "id": game_id
                    })
                    game_id += 1
            final_time = timeit.default_timer() - start_time
            print("[PLAY] Done fetching in %.3f seconds, average duration:" " %.3f seconds" % ((final_time, final_time / SELF_PLAY_MATCH)))
        finally:
            queue.close()
            results.close()


def play(agent):
    """ Game between two players, for evaluation """

    # Create the evaluation match queue of processes
    queue, results = create_episodes(deepcopy(agent), cores=PARALLEL_EVAL, match_number=EVAL_MATCHES)
    try:
        queue.join()

        # Gather the results and push them into a result queue that will be sent back to the evaluation process
        print("[EVALUATION] Starting to fetch fresh games")
        final_result = []
        for idx in range(EVAL_MATCHES):
            result = results.get()
            if result:
                final_result.append(pickle.loads(result))
        print("[EVALUATION] Done fetching")
    finally:
        queue.close()
        results.close()
    return final_result

