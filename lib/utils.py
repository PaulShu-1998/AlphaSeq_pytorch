import os
import torch
from models.agent import Agent
import numpy as np


def feature_extractor(state, q):
    feature = []
    for i in range(q):
        feature.append((2 ** i & (1 << state)) // 2 ** i)
    return np.array(feature)


def _prepare_state(state):
    """
    Transform the numpy state into a PyTorch tensor with cuda if available
    """
    x = torch.from_numpy(state).cuda()
    return x


def get_version(folder_path, version):
    """ Either get the last version ration of
        the specific folder or verify it version exists """

    if int(version) == -1:
        files = os.listdir(folder_path)
        if len(files) > 0:
            all_version = list(map(lambda x: int(x.split('-')[0]), files))
            all_version.sort()
            file_version = all_version[-1]
        else:
            return False
    else:
        test_file = "{}-extractor.pth.tar".format(version)
        if not os.path.isfile(os.path.join(folder_path, test_file)):
            return False
        file_version = version
    return file_version


def load_agent(folder, version):
    """ Load a player given a folder and a version """

    path = os.path.join('E:/Myproject/AlphaSeq_pytorch/', 'saved_models/')
    if folder == -1:
        folders = os.listdir(path)
        folders.sort()
        if len(folders) > 0:
            folder = folders[-1]
        else:
            return False, False
    elif not os.path.isdir(os.path.join(path, str(folder))):
        return False, False

    folder_path = os.path.join(path, str(folder))
    last_version = get_version(folder_path, version)
    if not last_version:
        return False, False

    return get_agent(folder, int(last_version))


def get_agent(current_time, version):
    """ Load the models of a specific player """

    path = os.path.join('E:/Myproject/AlphaSeq_pytorch/', 'saved_models/', str(current_time))
    try:
        mod = os.listdir(path)
        models = list(filter(lambda model: (model.split('-')[0] == str(version)), mod))
        models.sort()
        if len(models) == 0:
            return False, version
    except FileNotFoundError:
        return False, version

    agent = Agent()
    checkpoint = agent.load_models(path, models)
    return agent, checkpoint
