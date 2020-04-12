import torch
import numpy as np
import pickle
import time
from lib.process import MyPool
from lib.dataset import SelfPlayDataset
from lib.evaluate import evaluate
from lib.utils import get_agent, feature_extractor
from copy import deepcopy
from pymongo import MongoClient
from torch.utils.data import DataLoader
from const import MOVES, MAX_REPLACEMENT, LR, LR_DECAY, LR_DECAY_ITE, DEVICE, L2_REG, MOMENTUM, ADAM, BATCH_SIZE, TRAIN_STEPS, LOSS_TICK, REFRESH_TICK, q
from models.agent import Agent


class AlphaLoss(torch.nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_reward
    v : reward
    pi : self_play_probs
    p : probs

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, reward, self_play_reward, probs, self_play_probs):
        value_error = (self_play_reward - reward.view(-1)) ** 2
        policy_error = torch.sum((-self_play_probs * (1e-6 + probs).log()), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
        return total_error


def fetch_new_games(collection, dataset, last_id, loaded_version=None):
    """ Update the dataset with new episodes from the database """

    # Fetch new episodes in reverse order so we add the newest games first
    new_episodes = list(collection.find({"id": {"$gt": last_id}}).sort('_id', -1))
    added_moves = 0
    added_episodes = 0
    print("[TRAIN] Fetching: %d new episodes from the db" % (len(new_episodes)))

    for game in new_episodes:
        number_moves = dataset.update(pickle.loads(game['game']))
        added_moves += number_moves
        added_episodes += 1

        # You can't replace more than 40% of the dataset at a time
        if added_moves >= MOVES * MAX_REPLACEMENT and not loaded_version:
            break

    print("[TRAIN] Last id: %d, added episodes: %d, added moves: %d" % (last_id, added_episodes, added_moves))
    return last_id + added_episodes


def train_epoch(agent, optimizer, example, criterion):
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()
    reward, probs = agent.predict_train(example['state'])

    loss = criterion(reward, example['reward'], probs, example['probs'])
    loss.backward()
    optimizer.step()

    return float(loss)


def update_lr(lr, optimizer, total_ite, lr_decay=LR_DECAY, lr_decay_ite=LR_DECAY_ITE):
    """ Decay learning rate by a factor of lr_decay every lr_decay_ite iteration """

    if total_ite % lr_decay_ite != 0 or lr <= 0.0001:
        return lr, optimizer

    print("[TRAIN] Decaying the learning rate !")
    lr = lr * lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, optimizer


def create_state(current_version, lr, total_ite, optimizer):
    """ Create a checkpoint to be saved """

    state = {
        'version': current_version,
        'lr': lr,
        'total_ite': total_ite,
        'optimizer': optimizer.state_dict()
    }
    return state


def collate_fn(example):
    """ Custom way of collating example in dataloader """

    state = []
    probs = []
    rewards = []

    for ex in example:
        state.append(feature_extractor(ex[0], q))
        probs.append(ex[1])
        rewards.append(ex[2])

    state = torch.tensor(state, dtype=torch.float, device=DEVICE)
    probs = torch.tensor(probs, dtype=torch.float, device=DEVICE)
    winner = torch.tensor(rewards, dtype=torch.float, device=DEVICE)
    return state, probs, winner


def create_optimizer(agent, lr, param=None):
    """ Create or load a saved optimizer """

    joint_params = list(agent.extractor.parameters()) + list(agent.policy_net.parameters()) + list(agent.value_net.parameters())

    if ADAM:
        opt = torch.optim.Adam(joint_params, lr=lr, weight_decay=L2_REG)
    else:
        opt = torch.optim.SGD(joint_params, lr=lr, weight_decay=L2_REG, momentum=MOMENTUM)

    if param:
        opt.load_state_dict(param)

    return opt


def train(current_time, loaded_version):
    """ Train the models using the data generated by the self-play """

    last_id = -1
    total_ite = 1
    lr = LR
    version = 1
    pool = False
    criterion = AlphaLoss()
    dataset = SelfPlayDataset()

    # Database connection
    client = MongoClient()
    collection = client.alphaSeq[current_time]

    # First player either from disk or fresh
    if loaded_version:
        agent, checkpoint = get_agent(current_time, loaded_version)
        optimizer = create_optimizer(agent, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite']
        lr = checkpoint['lr']
        version = checkpoint['version']
        last_id = collection.count_documents({})-33
        agent.to_device()
    else:
        agent = Agent()
        optimizer = create_optimizer(agent, lr)
        state = create_state(version, lr, total_ite, optimizer)
        agent.save_models(state, current_time)
        agent.to_device()
    best_agent = deepcopy(agent)

    # Callback after the evaluation is done, must be a closure
    def new_agent(result):
        if result:
            nonlocal version, pending_agent, current_time, lr, total_ite, best_agent
            version += 1
            state = create_state(version, lr, total_ite, optimizer)
            best_agent = pending_agent
            pending_agent.save_models(state, current_time)
            print("[EVALUATION] New best player saved !")
        else:
            nonlocal last_id
            # Force a new fetch in case the player didnt improve
            last_id = fetch_new_games(collection, dataset, last_id)

    # Wait before the circular before is full
    while len(dataset) < MOVES:
        last_id = fetch_new_games(collection, dataset, last_id, loaded_version=loaded_version)
        time.sleep(2)

    print("[TRAIN] Circular buffer full !")
    print("[TRAIN] Starting to train !")
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE)

    while True:
        batch_loss = []
        for batch_idx, (state, move, reward) in enumerate(dataloader):
            running_loss = []
            lr, optimizer = update_lr(lr, optimizer, total_ite)

            # Evaluate a copy of the current network asynchronously
            if total_ite % TRAIN_STEPS == 0:
                pending_agent = deepcopy(agent)
                last_id = fetch_new_games(collection, dataset, last_id)

                # Wait in case an evaluation is still going on
                if pool:
                    print("[EVALUATION] Waiting for eval to end before re-eval")
                    pool.close()
                    pool.join()
                pool = MyPool(1)
                try:
                    pending_agent.to_cpu()
                    best_agent.to_cpu()
                    pool.apply_async(evaluate, args=(pending_agent, best_agent), callback=new_agent)
                except Exception as e:
                    client.close()
                    pool.terminate()

            example = {
                'state': state,
                'reward': reward,
                'probs': move
            }
            agent.to_device()
            loss = train_epoch(agent, optimizer, example, criterion)
            running_loss.append(loss)

            # Print running loss
            if total_ite % LOSS_TICK == 0:
                print("[TRAIN] current iteration: %d, averaged loss: %.3f" % (total_ite, np.mean(running_loss)))
                batch_loss.append(np.mean(running_loss))
                running_loss = []

            # Fetch new games
            if total_ite % REFRESH_TICK == 0:
                last_id = fetch_new_games(collection, dataset, last_id)

            total_ite += 1

        if len(batch_loss) > 0:
            print("[TRAIN] Average backward pass loss : %.3f, current lr: %f" % (np.mean(batch_loss), lr))

