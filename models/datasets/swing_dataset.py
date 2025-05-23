import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class SwingDataset(Dataset):
    def __init__(self, csv_path, session_col='session_swing'):
        self.data = pd.read_csv(csv_path)
        self.session_col = session_col

        # Group by session
        self.sessions = []
        for _, session_df in self.data.groupby(self.session_col):
            session_df = session_df.sort_values('timestep')
            self.sessions.append(session_df)

        # Build (state, next_state) pairs
        self.pairs = []
        for session in self.sessions:
            states = session.drop([self.session_col, 'timestep'], axis=1).values
            for i in range(len(states) - 1):
                state = states[i]
                next_state = states[i + 1]
                self.pairs.append((state, next_state))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        state, next_state = self.pairs[idx]
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        return state, next_state

# Example usage:
# dataset = SwingDataset(csv_path='rh_swing_data.csv')