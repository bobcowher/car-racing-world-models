import os
import random
import datetime
import cv2
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard.writer import SummaryWriter
from buffer import ReplayBuffer
from models.world_model import WorldModel
from models.q_model import QModel


class Agent:

    def __init__(self, env: gym.Env,
                       max_buffer_size: int = 100000,
                       target_update_interval: int = 1000,
                       dyna_k: int = 5) -> None:
        self.env    = env
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dyna_k = dyna_k

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("runs", exist_ok=True)

        obs, _ = self.env.reset()
        obs = self.process_observation(obs)

        self.buffer = ReplayBuffer(
            max_size=max_buffer_size,
            input_shape=obs.shape,
            n_actions=self.env.action_space.n,
            input_device=self.device,
            output_device=self.device,
        )

        self.world_model = WorldModel(
            observation_shape=obs.shape,
            embed_dim=512,
            n_actions=self.env.action_space.n,
        ).to(self.device)

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=0.0001)

        self.q_model = QModel(
            action_dim=self.env.action_space.n,
            hidden_dim=256,
            embed_dim=self.world_model.embed_dim,
        ).to(self.device)
        self.target_q_model = QModel(
            action_dim=self.env.action_space.n,
            hidden_dim=256,
            embed_dim=self.world_model.embed_dim,
        ).to(self.device)
        self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.0001)

        self.gamma    = 0.99
        self.epsilon  = 1.0
        self.min_epsilon   = 0.1
        self.epsilon_decay = 0.98

        self.target_update_interval = target_update_interval
        self.total_steps = 0

    def process_observation(self, obs):
        obs = cv2.resize(obs, (96, 96), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(obs).permute(2, 0, 1)  # (3, H, W) uint8

    def _encode_batch(self, obs_uint8):
        """obs_uint8: (B, C, H, W) float32 tensor → latents: (B, embed_dim)"""
        with torch.no_grad():
            return self.world_model.encode(obs_uint8 / 255.0)

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            latent = self.world_model.encode(
                obs.unsqueeze(0).float().to(self.device) / 255.0
            )  # (1, embed_dim)
            return self.q_model(latent).argmax(dim=1).item()

    def _dqn_update(self, states, actions, rewards, next_states, dones):
        actions = actions.unsqueeze(1).long()
        rewards = rewards.unsqueeze(1)
        dones   = dones.unsqueeze(1).float()

        q_sa = self.q_model(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_model(next_states).argmax(dim=1, keepdim=True)
            next_q       = self.target_q_model(next_states).gather(1, next_actions)
            targets      = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_sa, targets)
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.total_steps += 1
        return loss.item()

    def train_world_model(self, batch_size):
        obs, actions, rewards, next_obs, dones = self.buffer.sample_buffer(batch_size)
        loss, loss_dict = self.world_model.compute_loss(obs, actions, rewards, next_obs, dones)
        self.wm_optimizer.zero_grad()
        loss.backward()
        self.wm_optimizer.step()
        return loss_dict

    def train_dqn_real(self, batch_size):
        """One DQN update on real transitions (encoded on the fly)."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample_buffer(batch_size)
        states      = self._encode_batch(obs)
        next_states = self._encode_batch(next_obs)
        return self._dqn_update(states, actions, rewards, next_states, dones)

    def train_dqn_dyna(self, batch_size):
        """One DQN update on synthetic (imagined) transitions from the world model."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample_buffer(batch_size)
        action_onehot = F.one_hot(actions.long(), num_classes=self.env.action_space.n).float()

        with torch.no_grad():
            latents = self._encode_batch(obs)
            next_latents, reward_pred, done_pred = self.world_model.step(latents, action_onehot)
            synth_rewards = reward_pred.squeeze(-1) * 100.0
            synth_dones   = (done_pred.squeeze(-1) > 0.5).float()

        return self._dqn_update(latents, actions, synth_rewards, next_latents, synth_dones)

    def save(self):
        self.world_model.save_the_model("world_model", verbose=True)
        self.q_model.save_the_model("q_model", verbose=True)

    def load(self):
        self.world_model.load_the_model("world_model", device=self.device)
        self.q_model.load_the_model("q_model", device=self.device)
        self.target_q_model.load_the_model("q_model", device=self.device)

    def test(self, episodes=10):
        self.q_model.eval()
        total_rewards = []
        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs    = self.process_observation(obs)
            done   = False
            episode_reward = 0.0
            while not done:
                with torch.no_grad():
                    latent = self.world_model.encode(
                        obs.unsqueeze(0).float().to(self.device) / 255.0
                    )
                    action = self.q_model(latent).argmax(dim=1).item()
                next_obs, reward, term, trunc, _ = self.env.step(action)
                obs = self.process_observation(next_obs)
                done = term or trunc
                episode_reward += reward
            total_rewards.append(episode_reward)
            print(f"Test episode {episode} | reward: {episode_reward:.1f}")
        avg = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg:.1f}")
        self.q_model.train()
        return total_rewards

    def train(self, episodes=1200, wm_batch_size=32, dqn_batch_size=64,
              wm_updates=5, warmup=25):
        run_tag = f'dyna_k{self.dyna_k}_wmbs{wm_batch_size}_dqnbs{dqn_batch_size}'
        writer  = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{run_tag}')

        for episode in range(episodes):
            obs, _ = self.env.reset()
            obs    = self.process_observation(obs)
            done   = False
            episode_reward = 0.0
            episode_steps  = 0

            while not done:
                action = self.select_action(obs)
                next_obs, reward, term, trunc, _ = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                done     = term or trunc

                self.buffer.store_transition(obs, action, reward, next_obs, done)
                episode_reward += reward
                episode_steps  += 1
                obs = next_obs

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            wm_loss = dqn_real_loss = dqn_dyna_loss = 0.0

            if self.buffer.can_sample(wm_batch_size):
                for _ in range(wm_updates):
                    ld = self.train_world_model(wm_batch_size)
                wm_loss = ld["total"]

                if episode >= warmup:
                    dqn_real_loss = self.train_dqn_real(dqn_batch_size)
                    for _ in range(self.dyna_k):
                        dqn_dyna_loss += self.train_dqn_dyna(dqn_batch_size)
                    dqn_dyna_loss /= self.dyna_k

            print(f"Episode {episode} | reward: {episode_reward:.1f} | epsilon: {self.epsilon:.3f} | steps: {episode_steps}")

            writer.add_scalar("Train/episode_reward",    episode_reward,  episode)
            writer.add_scalar("Train/epsilon",           self.epsilon,    episode)
            writer.add_scalar("Train/dqn_real_loss",     dqn_real_loss,   episode)
            writer.add_scalar("Train/dqn_dyna_loss",     dqn_dyna_loss,   episode)
            writer.add_scalar("World Model/total_loss",  wm_loss,         episode)

            if episode % 10 == 0:
                self.save()
