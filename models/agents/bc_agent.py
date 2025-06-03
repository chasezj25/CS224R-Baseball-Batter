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

    def compute_loss(self, pred, targets):
        return self.loss_fn(pred, targets)

    def update(self, obs_batch, target_batch, optimizer):
        """
        Update policy using a batch of (obs, next_pose).
        """
        pred = self.policy(obs_batch)              # shape: (batch, action_dim)
        loss = self.compute_loss(pred, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, obs):
        with torch.no_grad():
            return self.policy(obs)
