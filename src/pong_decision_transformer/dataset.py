import numpy as np
import torch
from torch.utils.data import Dataset
import minari

def discount_cumsum(x, gamma):
    """Calculates discounted return."""
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret

class MinariDataset(Dataset):
    def __init__(self, env_name="mujoco/halfcheetah/medium-v0", context_length=20, gamma=1.0):
        self.context_length = context_length
        self.env_name = env_name
        
        print(f"Loading Minari dataset for {env_name}...")
        dataset = minari.load_dataset(env_name, download=True)
        
        # Segment dataset into trajectories based on episodes
        self.trajectories = []
        
        for episode in dataset:
            # Extract trajectory slice
            # minari observations include terminal state, so we take [:-1]
            traj_obs = episode.observations[:-1]
            traj_act = episode.actions
            traj_rew = episode.rewards
            
            # Calculate RTG
            traj_rtg = discount_cumsum(traj_rew, gamma=gamma)
            
            self.trajectories.append({
                'observations': traj_obs,
                'actions': traj_act,
                'rewards': traj_rew,
                'rtg': traj_rtg,
                'timesteps': np.arange(len(traj_obs))
            })

        print(f"Loaded {len(self.trajectories)} trajectories.")
        
        # Calculate global state mean and std for normalization (CRITICAL for MuJoCo DT)
        all_obs = np.concatenate([t['observations'] for t in self.trajectories])
        self.state_mean = np.mean(all_obs, axis=0)
        self.state_std = np.std(all_obs, axis=0) + 1e-6 # Add small epsilon to prevent div/0
        
        # Normalize all states in memory
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = len(traj['actions'])
        
        if traj_len >= self.context_length:
            start_idx = np.random.randint(0, traj_len - self.context_length + 1)
            end_idx = start_idx + self.context_length
        else:
            start_idx = 0
            end_idx = traj_len

        s = traj['observations'][start_idx:end_idx]
        a = traj['actions'][start_idx:end_idx]
        rtg = traj['rtg'][start_idx:end_idx]
        t = traj['timesteps'][start_idx:end_idx]

        pad_len = self.context_length - len(s)
        mask = np.concatenate([np.zeros(pad_len), np.ones(len(s))])

        if pad_len > 0:
            s = np.concatenate([np.zeros((pad_len, *s.shape[1:]), dtype=np.float32), s], axis=0)
            a = np.concatenate([np.zeros((pad_len, *a.shape[1:]), dtype=np.float32), a], axis=0)
            rtg = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg], axis=0)
            t = np.concatenate([np.zeros(pad_len, dtype=np.int64), t], axis=0)

        return {
            'states': torch.tensor(s, dtype=torch.float32),
            'actions': torch.tensor(a, dtype=torch.float32),
            'rtg': torch.tensor(rtg, dtype=torch.float32).unsqueeze(-1),
            'timesteps': torch.tensor(t, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }
