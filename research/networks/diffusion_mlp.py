# some code from https://github.com/SudeepDasari/dit-policy/blob/main/data4robotics/models/diffusion.py
from functools import partial
from typing import Optional

import gym
import torch
from torch import nn
import math
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .common import MLP
from .mlp import weight_init


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionMLPActor(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        ortho_init: bool = False,
        output_gain: Optional[float] = None,
        chunk_sz: int= 16,
        time_dim: int = 32,
        diffusion_train_steps: int = 32,
        diffusion_eval_steps: int = 32,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str ="squaredcos_cap_v2",
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1

        self.chunk_sz = chunk_sz
        self.observation_space = observation_space
        self.action_space = action_space
        self.temp_layers = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.act_dim = action_space.shape[0]
        out_dim = self.act_dim * chunk_sz
        in_dim = observation_space.shape[0] + time_dim + self.act_dim * chunk_sz
        self.mlp = MLP(in_dim, out_dim, **kwargs)
        self.ortho_init = ortho_init
        self.output_gain = output_gain

        self.diffusion_eval_steps = diffusion_eval_steps
        self.diffusion_train_steps = diffusion_train_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=diffusion_train_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(
        self,
        obs: torch.Tensor,
        t: torch.Tensor | None = None,
        noisy_act: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is not None:
            t = self.temp_layers(t)
            x = torch.cat((obs, t, noisy_act), dim=-1)
            return self.mlp(x)
        else:
            return self.sample(obs)

    def add_noise(self, actions, noise, t):
        return self.diffusion_schedule.add_noise(actions, noise, t)

    def get_rand_timesteps(self, shape):
        return torch.randint(0, self.diffusion_train_steps, shape, dtype=torch.long)

    def sample(self, obs: torch.Tensor):
        device = obs.device
        noisy_act = torch.randn(obs.shape[:-1] + (self.act_dim * self.chunk_sz,))

        self.diffusion_schedule.set_timesteps(self.diffusion_eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )
        for t in self.diffusion_schedule.timesteps:
            batched_t = t.unsqueeze(0).expand(obs.shape[:-1]).to(device)
            batched_t = self.temp_layers(batched_t)
            x = torch.cat((obs, batched_t, noisy_act), dim=-1)
            noise_pred = self.mlp(x)

            # take diffusion step
            noisy_act = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=t, sample=noisy_act
            ).prev_sample

        # return final action post diffusion
        return noisy_act[..., :self.act_dim]
