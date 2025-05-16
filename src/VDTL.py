import copy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from lamb import Lamb
from .util import DEFAULT_DEVICE,update_exponential_moving_average,sample_batch


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class VDTLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, learning_rate, discount=0.99, alpha=0.005):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)

        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())

        self.qf1_optimizer = optimizer_factory(self.qf.q1.parameters())
        self.qf2_optimizer = optimizer_factory(self.qf.q2.parameters())


        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    
    
    def update(self, dataset, batch_size, DT_batch=None):  
        
        def loss_fn(a_hat_dist,a,attention_mask,entropy_reg,):
   
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()

            loss = -(log_likelihood)
            return (loss,  -log_likelihood,  entropy)
        
        
        
        self.qf.train()
        self.vf.train()
        self.policy.train()
        self.q_target.eval()

        logs = dict()

        for _ in range(100):
            batch = sample_batch(dataset, batch_size)
            observations = batch["observations"]
            actions = batch["actions"]
            next_observations = batch["next_observations"]
            rewards = batch["rewards"]
            terminals = batch["terminals"]

            with torch.no_grad():
                target_q = self.q_target(observations, actions)
                next_v = self.vf(next_observations)


            adv = target_q - self.vf(observations)
            v_loss = asymmetric_l2_loss(adv, self.tau)
            self.v_optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            self.v_optimizer.step()

            # Update Q function
            targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
            targets=targets.detach()
            qs = self.qf.both(observations, actions)
            qf1_loss = F.mse_loss(qs[0], targets)
            qf2_loss = F.mse_loss(qs[1], targets)

   
            self.qf1_optimizer.zero_grad(set_to_none=True)
            qf1_loss.backward()
            self.qf1_optimizer.step()
            self.qf2_optimizer.zero_grad(set_to_none=True)
            qf2_loss.backward()
            self.qf2_optimizer.step()

      
            update_exponential_moving_average(self.q_target, self.qf, self.alpha)



        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask, next_observations ,terminals= DT_batch
        

        # Update policy
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]

        state_preds, action_preds, reward_preds = self.policy.forward(
            states, actions, rewards, action_target, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        action_target_ = action_target.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1]>0].mean()

        bc_losses= F.mse_loss(action_preds_, action_target_) + states_loss




        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_loss = torch.mean(exp_adv * bc_losses)
        policy_loss = bc_losses

        
        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]


        actor_states = states.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        qs = self.qf.both(actor_states, action_preds_)
        if np.random.uniform() > 0.5:
            q_loss = - qs[0].mean() / qs[1].abs().mean().detach()
        else:
            q_loss = - qs[1].mean() / qs[0].abs().mean().detach()


        policy_loss = policy_loss + 0.5* q_loss
 
    

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()

        self.policy_optimizer.step()
        self.policy_lr_schedule.step()



        logs['BC Loss'] = bc_losses.item()
        logs['Actor Loss'] = policy_loss.item()
        logs['QL Loss'] = q_loss.item()
        
        return logs