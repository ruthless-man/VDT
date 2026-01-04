import random
import d4rl
import numpy as np
import torch
import torch.nn as nn
import gym
import copy

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DEFAULT_DEVICE = torch.device('cpu')

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.Mish, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x



def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}







def evaluate_policy(env, qf,policy, env_targets, num_eval_episodes,scale, state_dim, act_dim, max_ep_len,mode,state_mean,state_std):
    returns, lengths = [], []
    log = dict()
# for targets in env_targets:
    for _ in range(num_eval_episodes):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                policy,
                qf,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return= [t/scale for t in env_targets],
                state_mean=state_mean,
                state_std=state_std,
                mode=mode,
            )
        returns.append(ret)
        lengths.append(length)
    log[f'target_length_mean'] = np.mean(lengths)
    log[f'target_normalized_score'] = env.get_normalized_score(np.mean(returns)) * 100
    return log

        


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)



def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum




def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        q_network,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        target_return=None,
        mode='normal',
        device=DEFAULT_DEVICE,
        num_candidates=5,
        plan_horizon=5
    ):


    model.eval()
    q_network.eval()
    model.to(device=device)
    q_network.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)



    env_name = getattr(env, 'spec', None) and env.spec.id
    if env_name:

        original_sim_state = env.sim.get_state() if hasattr(env, 'sim') else None
    else:
        original_env = copy.deepcopy(env)  


    state = env.reset()

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32) 


    target_return= torch.tensor(target_return, device=device, dtype=torch.float32).unsqueeze(-1)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):
 
        remaining_steps = max_ep_len - t
        current_plan_horizon = min(plan_horizon, remaining_steps)
        
   
        if current_plan_horizon <= 0:
     
            with torch.no_grad():
                _, action, _= model.get_action(
                    states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions=actions.to(dtype=torch.float32),
                    returns_to_go=target_return[:, -1].unsqueeze(1).to(dtype=torch.float32),
                    timesteps=timesteps.to(dtype=torch.long))
            action = action.squeeze(0).cpu().numpy()
        else:

            candidate_actions = []
            for perturbed_rtg in target_return:
                _, action, _ = model.get_action(
                    states=(states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions=actions.to(dtype=torch.float32),
                    returns_to_go=perturbed_rtg.to(dtype=torch.float32),
                    timesteps=timesteps.to(dtype=torch.long),
                )
         
                action = action.clamp(*model.action_range)
                candidate_actions.append(action.squeeze(0))
            candidate_actions = torch.stack(candidate_actions)

   
            cumulative_q_values = []
            for action_index, action in enumerate(candidate_actions):
      
                if env_name:
                    temp_env = gym.make(env_name)
                    temp_env.reset()
                    temp_env.sim.set_state(original_sim_state)
                    temp_env.sim.forward()
                else:
                    temp_env = copy.deepcopy(original_env)
                    temp_env.reset()

                total_q = 0
                current_states = states.clone()
                current_actions = torch.cat([actions, action.unsqueeze(0)], dim=0)
                current_rtg = target_return[action_index].clone().unsqueeze(0)
                current_action = action.clone()

                for step in range(current_plan_horizon):
                    safe_timestep = min(t + step + 1, max_ep_len - 1)

                    with torch.no_grad():
                        q_val = q_network(
                            (current_states[-1:] - state_mean) / state_std,
                            current_action.unsqueeze(0)
                        )
                        total_q += q_val.item()

                    temp_state, temp_reward, temp_done, _ = temp_env.step(
                        current_action.cpu().numpy()
                    )
                    temp_state_tensor = torch.from_numpy(temp_state).to(device, dtype=torch.float32).unsqueeze(0)

                    new_rtg = (current_rtg[0, -1] - temp_reward / scale) / 0.99
                    current_rtg = torch.cat([current_rtg, new_rtg.reshape(1, 1)], dim=1)

                    with torch.no_grad():
                        _, next_action, _ = model.get_action(
                            states=torch.cat([current_states, temp_state_tensor], dim=0),
                            actions=current_actions,
                            returns_to_go=current_rtg,
                            timesteps=torch.cat([
                                timesteps,
                                torch.tensor([[safe_timestep]], device=device)
                            ], dim=1)
                        )
                    

                    next_action = next_action.clamp(*model.action_range).unsqueeze(0)
 
                    current_states = torch.cat([current_states, temp_state_tensor], dim=0)
                    current_actions = torch.cat([current_actions, next_action], dim=0)
                    current_action = next_action.squeeze(0)

                    if temp_done:
                        break

                cumulative_q_values.append(total_q)


            best_idx = torch.argmax(torch.tensor(cumulative_q_values))
            action = candidate_actions[best_idx].detach().cpu().numpy()


        state, reward, done, _ = env.step(action)
        

        cur_state = torch.from_numpy(state).to(device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        actions = torch.cat([actions, torch.tensor(action, device=device).reshape(1, -1)], dim=0)


        raw_rtg = target_return[:, -1].unsqueeze(1) - (reward / scale)
        pred_return = raw_rtg / 0.99
        target_return = torch.cat([target_return, pred_return], dim=1)


        safe_global_timestep = min(t + 1, max_ep_len - 1)
        timesteps = torch.cat([timesteps,torch.tensor([[safe_global_timestep]], device=device)], dim=1)
        
        episode_return += reward
        episode_length += 1
        
        if done:
            break

    return episode_return, episode_length








def vec_evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        target_return=None,
        mode='normal',
        device=DEFAULT_DEVICE,
        use_mean=True,
    ):

    num_envs=1
    model.eval()
    # q_network.eval()
    model.to(device=device)
    # q_network.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    states = (torch.from_numpy(state).reshape(num_envs, state_dim).to(device=device, dtype=torch.float32)).reshape(num_envs, -1, state_dim)
    next_states = torch.zeros(0, device=device, dtype=torch.float32)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)



    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(num_envs, -1, 1)
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(num_envs, -1)

    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)
    unfinished = np.ones(num_envs).astype(bool)

    for t in range(max_ep_len):

        actions = torch.cat([actions,
                             torch.zeros((num_envs, act_dim), device=device).reshape(num_envs, -1, act_dim)
                             ],dim=1)
        rewards = torch.cat([
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1)
            ],dim=1)

        _, action , _ = model.get_action(
            states=(states.to(dtype=torch.float32) - state_mean) / state_std,
            actions=actions,
            returns_to_go=target_return.to(dtype=torch.float32),
            timesteps=timesteps)



        action=action.reshape(num_envs, -1, act_dim)[:, -1]

        action = action.clamp(*model.action_range)

        state, reward, done, _ = env.step(action.detach().cpu().numpy())

        state=state.reshape(1, state_dim)
        reward=reward.reshape(1, 1)
        done=np.array([done])
        

        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action

        next_state = _['terminal_observation'] if ('terminal_observation' in _) else state
    
        state = (torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim))
        states = torch.cat([states, state], dim=1)
        next_state = (torch.from_numpy(next_state).to(device=device).reshape(num_envs, -1, state_dim))
        next_states = torch.cat([next_states.float(), next_state.float()], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        raw_rtg = target_return[:, -1].unsqueeze(1) - (reward / scale)
        pred_return = raw_rtg / 0.99
        target_return = torch.cat([target_return, pred_return.reshape(num_envs, -1, 1)], dim=1)


    
        
        timesteps = torch.cat([
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )* (t + 1),
            ],dim=1)


        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "next_observations": next_states[ii].detach().cpu().numpy()[:ep_len],
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len].reshape(-1),
            "terminals": terminals,
        }
        trajectories.append(traj)


    return (episode_return.reshape(num_envs),  episode_length.reshape(num_envs),  trajectories)