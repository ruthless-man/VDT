import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
from model import TrajectoryModel
from trajectory_gpt2 import GPT2Model
from torch import distributions as pyd



class DecisionTransformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            action_range,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            scale=1.,
            stochastic_policy=False,
            ordering=0,
            init_temperature=0.1,
            target_entropy=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.config = config
        self.scale = scale
        self.rtg_no_q= False
        self.infer_no_q= False
        self.ordering= ordering
        self.action_range=action_range
        
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)


        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        self.predict_action = nn.Sequential(
        *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else [])))

  
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)



    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None
        
    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
     
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)


        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        time_embeddings = self.embed_timestep(timesteps)



        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings



        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

  
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs,attention_mask=stacked_attention_mask,)
        x = transformer_outputs['last_hidden_state']


        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:, 1])
        state_preds = self.predict_state(x[:, 2])
        rewards_preds = None


        return state_preds, action_preds, rewards_preds


    def get_action(self, states, actions, returns_to_go=None, timesteps=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]


            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

          
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),device=actions.device), actions],dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],dim=1).to(dtype=torch.long)


        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
        
        return _ , action_preds[0,-1], _
    
    def clamp_action(self, action):
        return action.clamp(*self.action_range)
