import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 히든 레이어를 256으로 확장하여 복잡한 환경 인지능력 상승
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        mu = self.actor(state)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = 10
        self.batch_size = 64 # 한 번에 64개씩 잘라서 학습 (미니배치)
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.numpy()[0], log_prob.item(), value.item()

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs']))
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values']

        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        values = torch.FloatTensor(values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        # 수집된 데이터를 배치 단위로 쪼개어 학습 (그래프 수렴 안정화)
        for _ in range(self.k_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                dist, state_values = self.policy(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(state_values.squeeze(), batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5) # 그라디언트 폭주 방지
                self.optimizer.step()