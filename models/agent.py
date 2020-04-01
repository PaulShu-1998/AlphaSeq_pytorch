import os
from .feature import Extractor
from .policy import PolicyNet
from .value import ValueNet
from const import INPLANES, OUTPLANES, DEVICE, OUTPLANES_MAP
import torch


class Agent:
    def __init__(self):
        """Create an agent and initialize the networks"""

        self.extractor = Extractor(INPLANES, OUTPLANES_MAP).to(DEVICE)
        self.value_net = ValueNet(OUTPLANES_MAP).to(DEVICE)
        self.policy_net = PolicyNet(OUTPLANES_MAP).to(DEVICE)

    def predict(self, state):
        feature_maps = self.extractor(state)
        value = self.value_net(feature_maps)
        probs = self.policy_net(feature_maps)
        return value, probs

    def save_models(self, state, current_time):
        for model in ["extractor", "policy_net", "value_net"]:
            self._save_checkpoint(getattr(self, model), model, state, current_time)

    def _save_checkpoint(self, model, filename, state, current_time):
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'saved_models', current_time)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        filename = os.path.join(dir_path, "{}-{}.pth.tar".format(state['version'], filename))
        state['model'] = model.state_dict()
        torch.save(state, filename)

    def load_models(self, path, models):
        names = ["extractor", "policy_net", "value_net"]
        for i in range(len(models)):
            checkpoint = torch.load(os.path.join(path, models[i]))
            model = getattr(self, names[i])
            model.load_state_dict(checkpoint['model'])
            return checkpoint
