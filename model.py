import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from src.util import discount_cumsum
from src.util import DEFAULT_DEVICE

class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
     
        return None, None, None

    def get_action(self, states, actions, rewards, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(actions[-1])
    







def get_batch_online(trajectories, max_len, K, pct_traj , env_name, state_dim, act_dim, max_ep_len, scale, batch_size=256):

    states, traj_lens, returns , next_observations = [], [], [], []

    for path in trajectories:
        states.append(path['observations'])
        next_observations.append(path['next_observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    next_observations = np.concatenate(next_observations, axis=0)
    next_observations_mean, next_observations_std = np.mean(next_observations, axis=0), np.std(next_observations, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

 
    
    num_timesteps = max(int(pct_traj*num_timesteps), 1)

    sorted_inds = np.argsort(returns)  
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    '''---------------------------------------------------------------------------------------------------------------'''


    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    s, a, r, d, rtg, timesteps, mask, target_a,next_observations,terminals = [], [], [], [], [], [], [], [],[],[]
    
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1)
        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        next_observations.append(traj['next_observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        target_a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
        else:

            d.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
        terminals.append(traj['terminals'][si:si + max_len].reshape(1, -1, 1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

        traj_rewards = traj['rewards']
        r.append(traj_rewards[si:si + max_len].reshape(1, -1, 1))
        rtg.append(discount_cumsum(traj_rewards[si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
        
 
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        next_observations[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), next_observations[-1]], axis=1)
        next_observations[-1] = (next_observations[-1] - next_observations_mean) / next_observations_std

        a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        target_a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), target_a[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen, 1)), d[-1]], axis=1)
        terminals[-1]=np.concatenate([np.ones((1, max_len - tlen, 1)), terminals[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)
    next_observations = torch.from_numpy(np.concatenate(next_observations, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)

    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)
    target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long,device=DEFAULT_DEVICE)
    terminals=torch.from_numpy(np.concatenate(terminals,axis=0)).to(dtype=torch.bool,device=DEFAULT_DEVICE)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32,device=DEFAULT_DEVICE)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long,device=DEFAULT_DEVICE)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=DEFAULT_DEVICE)

    return s, a, r, target_a, d, rtg, timesteps, mask,next_observations,terminals




