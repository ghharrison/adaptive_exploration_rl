from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from torch.nn import init
from config import config


class EmbeddingModel(nn.Module):
    def __init__(self, obs_size, num_outputs, use_rnd=False):
        super(EmbeddingModel, self).__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.last = nn.Linear(32 * 2, num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.rnd = None
        if use_rnd:
            self.rnd = RNDModel(32, num_outputs)
            self.rnd_optimizer = optim.Adam(self.rnd.parameters(), lr=1e-3) # consider 5e-4 or 5e-3
            self.rnd_error_memory = []
        self.L = 5 # maximum episodic intrinsic reward scaling


    def forward(self, x1, x2):
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.last(x)
        return nn.Softmax(dim=2)(x)

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def train_model(self, batch):
        batch_size = torch.stack(batch.state).size()[0]
        # last 5 in sequence
        states = torch.stack(batch.state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        next_states = torch.stack(batch.next_state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        actions = torch.stack(batch.action).view(batch_size, config.sequence_length, -1).long()[:, -5:, :]

        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        actions_one_hot = torch.squeeze(F.one_hot(actions, self.num_outputs)).float()
        loss = nn.MSELoss()(net_out, actions_one_hot)
        loss.backward()
        self.optimizer.step()
        if self.rnd is not None:
            self.rnd_optimizer.step()
        return loss.item()


    def compute_intrinsic_reward(
        self,
        episodic_memory: List,
        current_c_state: Tensor,
        k=10,
        kernel_cluster_distance=0.008,
        kernel_epsilon=0.0001,
        c=0.001,
        sm=8,
    ) -> float:
        state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
        state_dist.sort(key=lambda x: x[1])
        state_dist = state_dist[:k]
        dist = [d[1].item() for d in state_dist]
        dist = np.array(dist)

        # TODO: moving average
        dist = dist / np.mean(dist)


        # Add life-long curiosity factor from RND
        alpha = 1
        lifelong_intrinsic_reward = 0
        if self.rnd is not None:
            current_c_state_reshape = torch.reshape(current_c_state, (1, 1, -1))
            predict_feature, target_feature = self.rnd(current_c_state_reshape)
            lifelong_intrinsic_reward = ((target_feature - predict_feature).pow(2).sum(1) / 2).item()
            # print("shape of rnd error: ", lifelong_intrinsic_reward.shape)
            
            self.rnd_error_memory.append(lifelong_intrinsic_reward)
            
            sigma_e = np.std(self.rnd_error_memory)
            mu_e = np.mean(self.rnd_error_memory)
            alpha = 1 + (lifelong_intrinsic_reward - mu_e) / sigma_e
            # print("TEST - alpha:", alpha)
            # alpha=1




        dist = np.max(dist - kernel_cluster_distance, 0)
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        s = np.sqrt(np.sum(kernel)) + c

        if np.isnan(s) or s > sm:
            episodic_intrinsic_reward = 0
        else:
            episodic_intrinsic_reward = 1 / s

        return episodic_intrinsic_reward * np.minimum(np.maximum(alpha, 1), self.L), lifelong_intrinsic_reward


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 64 # 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature