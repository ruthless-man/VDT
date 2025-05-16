import os

import gym
import numpy as np
import torch
from tqdm import trange
import random
import wandb
from src.VDTL import VDTLearning
from src.value_functions import TwinQ, ValueFunction
from src.util import set_seed, torchify, evaluate_policy, discount_cumsum, vec_evaluate_episode_rtg
from decision_transformer import DecisionTransformer
from lamb import Lamb
from replay_buffer import ReplayBuffer
from model import get_batch_online


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ESSENTIAL_KEYS = {
    'observations', 'actions', 'rewards',
    'next_observations', 'terminals', 'timeouts'
}

ESSENTIAL_KEYS_1 = {
    'observations', 'actions', 'rewards',
    'next_observations', 'terminals'
}



def split_into_trajectories(dataset):
    trajectories = []
    current_traj = {k: [] for k in ESSENTIAL_KEYS}
    for i in range(len(dataset['observations'])):
        for k in ESSENTIAL_KEYS:
            current_traj[k].append(dataset[k][i])
        if dataset['terminals'][i] or dataset['timeouts'][i]:
            traj = {k: np.array(v) for k, v in current_traj.items()}
            trajectories.append(traj)
            current_traj = {k: [] for k in ESSENTIAL_KEYS}
    if len(current_traj['observations']) > 0: 
        traj = {k: np.array(v) for k, v in current_traj.items()}
        trajectories.append(traj)
    return trajectories





def process_qlearning_dataset(trajectories,time_out=True):
    total_len = sum(len(t['observations']) for t in trajectories)

    q_data = {}
    if time_out:
        for k in ESSENTIAL_KEYS:
            dtype = np.float32  
            if k in ['terminals', 'timeouts']:  
                dtype = np.bool_
            q_data[k] = np.empty((total_len, *trajectories[0][k].shape[1:]), dtype=dtype)
        index = 0
        for traj in trajectories:
            traj_len = len(traj['observations'])
            for k in ESSENTIAL_KEYS:
                q_data[k][index:index+traj_len] = traj[k]
            index += traj_len
        return q_data
    else:
        for k in ESSENTIAL_KEYS_1:
            dtype = np.float32  
            if k in ['terminals', 'timeouts']:  
                dtype = np.bool_
            q_data[k] = np.empty((total_len, *trajectories[0][k].shape[1:]), dtype=dtype)
        index = 0
        for traj in trajectories:
            traj_len = len(traj['observations'])
            for k in ESSENTIAL_KEYS_1:
                q_data[k][index:index+traj_len] = traj[k]
            index += traj_len
        return q_data





def get_env_and_dataset(env_name,scale,raw_dataset,state_mean, state_std):
    env = gym.make(env_name)

    filtered_dataset = {k: raw_dataset[k] for k in ESSENTIAL_KEYS}
    
    trajectories = split_into_trajectories(filtered_dataset)

    dataset = process_qlearning_dataset(trajectories)
    

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):

        dataset['rewards'] /= scale
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.0

    dataset['observations'] = (dataset['observations'] - state_mean) / state_std
    
    return env, {k: torchify(v) for k, v in dataset.items() if k != "timeouts"}





def main(args):
    
    if 'hopper' in args.env_name:
        max_ep_len = 1000
        env_targets = [72000,36000, 18000, 7200, 3600, 1800,720]  
        scale = 1000.  # normalization for rewards/returns
    elif 'halfcheetah' in args.env_name:
        max_ep_len = 1000
        env_targets = [12000, 9000, 6000]
        scale = 1000.
    elif 'walker2d' in args.env_name:
        max_ep_len = 1000
        env_targets = [5000, 4000, 2500]
        scale = 1000.
    elif 'reacher2d' in args.env_name:
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif 'pen' in args.env_name:
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif 'hammer' in args.env_name:
        max_ep_len = 1000
        env_targets = [12000, 6000, 3000]
        scale = 1000.
    elif 'door' in args.env_name:
        max_ep_len = 1000
        env_targets = [2000, 1000, 500]
        scale = 100.
    elif 'kitchen' in args.env_name:
        max_ep_len = 1000
        env_targets = [500, 250]
        scale = 100.
    elif 'maze2d' in args.env_name:
        max_ep_len = 1000
        env_targets = [300, 200, 150,  100, 50, 20]
        scale = 10.
    elif 'antmaze' in args.env_name:
        max_ep_len = 1000
        env_targets = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
        scale = 1.



    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    raw_dataset = env.get_dataset()

    if 'next_observations' not in raw_dataset:
        observations = raw_dataset['observations']
        terminals = raw_dataset['terminals']
        timeouts = raw_dataset['timeouts']
        next_observations = np.empty_like(observations)

        for i in range(len(observations) - 1):
            if terminals[i] or timeouts[i]:
                next_observations[i] = observations[i]  
            else:
                next_observations[i] = observations[i + 1]


        next_observations[-1] = observations[-1]  

        raw_dataset['next_observations'] = next_observations


    filtered_dataset = {k: raw_dataset[k] for k in ESSENTIAL_KEYS}#
    trajectories = split_into_trajectories(filtered_dataset)





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
    print(f'Starting new experiment: {args.env_name}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

    K = args.K
    pct_traj = args.pct_traj
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

    offline_trajs = [trajectories[ii] for ii in sorted_inds]


    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  
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





    torch.set_num_threads(1)
    env, dataset = get_env_and_dataset(args.env_name,scale,raw_dataset, state_mean, state_std)

    action_range = [float(env.action_space.low.min()) + 1e-6,float(env.action_space.high.max()) - 1e-6,]
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   
    set_seed(args.seed, env=env)

    target_entropy=-act_dim

    policy= DecisionTransformer(
    state_dim=obs_dim,
    act_dim=act_dim,
    max_length=args.K,
    max_ep_len=max_ep_len,
    hidden_size=args.embed_dim,
    action_range=action_range,
    n_layer=args.n_layer,
    n_head=args.n_head,
    n_inner=4*args.embed_dim,
    activation_function=args.activation_function,
    n_positions=1024,
    resid_pdrop=args.dropout,
    attn_pdrop=args.dropout,
    scale=scale,

    stochastic_policy=False,
    ordering=args.ordering,
    init_temperature=args.init_temperature,
    target_entropy=target_entropy
    
    )
    

    qf= TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)


    
    total_params = sum(p.numel() for p in qf.parameters() ) + sum(p.numel() for p in vf.parameters()) + sum(p.numel() for p in policy.parameters())

    print(f"Total parameters (trainable): {total_params:,}")




    VDT = VDTLearning(
        qf=qf,
        vf=vf,
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
        learning_rate=args.learning_rate
    )

    
    for step in trange(args.n_steps):

        outputs=VDT.update(dataset, args.batch_size, DT_batch=get_batch(batch_size=args.batch_size,max_len=args.K))
        if (step+1) % args.eval_period == 0:
            a=evaluate_policy(env, qf,policy, env_targets, args.n_eval_episodes,scale,state_dim,act_dim,max_ep_len,args.mode, 
                            state_mean, state_std)
            outputs.update(a)

            if args.save_checkpoint:

                save_dir = os.path.join(args.checkpoint_dir, args.env_name)
                os.makedirs(save_dir, exist_ok=True)

                torch.save(qf.state_dict(), os.path.join(save_dir, f"qf_{step+1}.pth"))
                torch.save(vf.state_dict(), os.path.join(save_dir, f"vf_{step+1}.pth"))
                torch.save(policy.state_dict(), os.path.join(save_dir, f"policy_{step+1}.pth"))
                print(f"Model saved at step {step+1} in {save_dir}")





    def augment_trajectories(env,replay_buffer):
        max_ep_len = 1000

        with torch.no_grad():

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                policy,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=7200/scale,
                mode="normal",
                state_mean=state_mean,
                state_std=state_std,
                device=DEFAULT_DEVICE,
                use_mean=False
            )

        replay_buffer.add_new_trajs(trajs)

        print("aug_traj/return", np.mean(returns)," \naug_traj/length", np.mean(lengths))
    


    replay_buffer = ReplayBuffer(args.replay_size, offline_trajs)
    
    if args.online_finetune:

        for online_iter in trange(args.max_online_iters):

            augment_trajectories(env=env, replay_buffer=replay_buffer)
            dataset = process_qlearning_dataset(replay_buffer.trajectories,time_out=False)
            if any(s in args.env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
                dataset['rewards'] /= scale
            elif 'antmaze' in args.env_name:
                dataset['rewards'] -= 1.0

            dataset['observations'] = (dataset['observations'] - state_mean) / state_std
            dataset= {k: torchify(v) for k, v in dataset.items() if k != "timeouts"}
            is_last_iter = online_iter == args.max_online_iters - 1

            outputs=VDT.update(dataset, args.batch_size, 
                               DT_batch=get_batch_online(trajectories=replay_buffer.trajectories, 
                                                         max_len=args.K, 
                                                         K=args.K, 
                                                         pct_traj = args.pct_traj, 
                                                         env_name=args.env_name, 
                                                         state_dim=state_dim,
                                                         act_dim= act_dim, 
                                                         max_ep_len=max_ep_len, 
                                                         scale=scale,
                                                         batch_size=args.batch_size))
            
            if (online_iter ) % args.eval_interval == 0 or is_last_iter:
                a=evaluate_policy(env, qf,policy, env_targets, args.n_eval_episodes,scale,state_dim,act_dim,max_ep_len,args.mode, 
                            state_mean, state_std)
                outputs.update(a)






if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default='antmaze-umaze-v0', type=str)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=3)
    parser.add_argument('--eval-period', type=int, default=100)
    parser.add_argument('--n-eval-episodes', type=int, default=5)


    '''在线微调'''
    parser.add_argument("--ordering", type=int, default=0)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--online_finetune", action="store_true", default=True)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--max_online_iters", type=int, default=25000)

    #用于DT的参数
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--reward_tune", default='no', type=str)
    parser.add_argument("--log_to_wandb", action="store_true",default=False)
    parser.add_argument('--mode', type=str, default='normal')

    #模型保存
    parser.add_argument('--checkpoint_dir', type=str, default='./determin_checkpoints', help='path to save/load checkpoints')
    parser.add_argument('--save_checkpoint', action='store_true', help='save model checkpoint')


    main(parser.parse_args())