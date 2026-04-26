import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.base import BaseModel
from models.dynamics_model import DynamicsModel


class WorldModel(BaseModel):

    def __init__(self, observation_shape=(), embed_dim=512, n_actions=4, embed_norm='layernorm'):
        super().__init__()

        self.encoder = Encoder(observation_shape=observation_shape, embed_dim=embed_dim)

        self.dynamics = DynamicsModel(embed_dim=embed_dim, n_actions=n_actions, hidden_dim=1024)

        if embed_norm == 'layernorm':
            self.embed_norm_layer = nn.LayerNorm(embed_dim)
        else:
            self.embed_norm_layer = None
        self.embed_norm_type = embed_norm

        self.reward_pred = nn.Linear(embed_dim + n_actions, 1)
        self.done_pred   = nn.Linear(embed_dim + n_actions, 1)

        self.embed_dim = embed_dim
        self.n_actions = n_actions

        print(f"WorldModel (Dyna): obs={observation_shape} embed={embed_dim} actions={n_actions}")

    def _norm(self, embed):
        if self.embed_norm_type == 'layernorm':
            return self.embed_norm_layer(embed)
        return embed

    def encode(self, obs):
        """obs: (B, C, H, W) float [0,1] → latent: (B, embed_dim)"""
        return self._norm(self.encoder(obs))

    def step(self, latent, action_onehot):
        """
        One-step latent prediction.
        latent:        (B, embed_dim)
        action_onehot: (B, n_actions)
        Returns: next_latent (B, embed_dim), reward (B, 1), done (B, 1)
        """
        next_latent = self._norm(self.dynamics(latent, action_onehot))
        embed_action = torch.cat([latent, action_onehot], dim=-1)
        reward = self.reward_pred(embed_action)
        done   = torch.sigmoid(self.done_pred(embed_action))
        return next_latent, reward, done

    def compute_loss(self, obs, actions, rewards, next_obs, dones):
        obs_norm      = obs.float() / 255.0
        next_obs_norm = next_obs.float() / 255.0

        action_onehot = F.one_hot(actions.long(), num_classes=self.n_actions).float()

        latents             = self.encode(obs_norm)
        next_latents_target = self.encode(next_obs_norm).detach()

        next_latents_pred, reward_pred, done_pred = self.step(latents, action_onehot)

        dynamics_loss = F.mse_loss(next_latents_pred, next_latents_target)
        reward_loss   = F.mse_loss(reward_pred.squeeze(-1), rewards.float() / 100.0)
        done_loss     = F.binary_cross_entropy(done_pred.squeeze(-1), dones.float())

        loss = dynamics_loss + 2.0 * reward_loss + 0.5 * done_loss

        return loss, {
            "total":    loss.item(),
            "dynamics": dynamics_loss.item(),
            "reward":   reward_loss.item(),
            "done":     done_loss.item(),
        }
