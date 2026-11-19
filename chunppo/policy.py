"""Feature extractor: NatureCNN(1024) on the Ultimate image + MLP on own-action history."""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class Extractor(BaseFeaturesExtractor):
    def __init__(self, space, cnn_dim=1024, hist_dim=128):
        super().__init__(space, features_dim=cnn_dim + hist_dim)
        self.cnn = NatureCNN(space['image'], features_dim=cnn_dim)
        self.hist = nn.Sequential(nn.Linear(space['hist'].shape[0], hist_dim), nn.ReLU())

    def forward(self, obs):
        return torch.cat([self.cnn(obs['image']), self.hist(obs['hist'])], dim=1)


POLICY_KWARGS = dict(features_extractor_class=Extractor, net_arch=[])
