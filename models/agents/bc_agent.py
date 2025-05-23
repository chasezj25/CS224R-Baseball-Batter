import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)

class BCAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        self.policy = BCPolicy(obs_dim, action_dim, hidden_dim)
        self.loss_fn = nn.MSELoss()
        self.dagger_buffer = []  # For DAgger: stores (obs, actions) tuples

    def compute_loss(self, pred, targets):
        # targets: (batch, num_experts, action_dim)
        # pred: (batch, action_dim)
        avg_targets = targets.mean(dim=1)
        return self.loss_fn(pred, avg_targets)

    def update(self, batch, optimizer):
        obs = batch['obs']  # shape: (batch, obs_dim)
        targets = batch['actions']  # shape: (batch, num_experts, action_dim)
        pred = self.policy(obs)
        loss = self.compute_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def add_dagger_data(self, obs, expert_actions):
        """
        Add new (obs, expert_actions) pairs to the DAgger buffer.
        obs: (batch, obs_dim)
        expert_actions: (batch, num_experts, action_dim)
        """
        self.dagger_buffer.append((obs.detach().cpu(), expert_actions.detach().cpu()))

    def get_dagger_dataset(self):
        """
        Returns all DAgger data as tensors.
        """
        if not self.dagger_buffer:
            return None, None
        obs_list, actions_list = zip(*self.dagger_buffer)
        obs = torch.cat(obs_list, dim=0)
        actions = torch.cat(actions_list, dim=0)
        return obs, actions

    def update_with_dagger(self, optimizer, batch_size=64):
        """
        Update the agent using the aggregated DAgger dataset.
        """
        obs, actions = self.get_dagger_dataset()
        if obs is None:
            return None
        num_samples = obs.size(0)
        idx = torch.randperm(num_samples)
        obs = obs[idx]
        actions = actions[idx]
        total_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_obs = obs[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            pred = self.policy(batch_obs)
            loss = self.compute_loss(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / (num_samples // batch_size)
