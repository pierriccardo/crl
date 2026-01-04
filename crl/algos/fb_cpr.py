import math
import copy
import json
import tyro
import wandb
import numbers
import dataclasses
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch import autograd
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from pathlib import Path
import safetensors.torch
from safetensors.torch import save_model as safetensors_save_model

from tqdm import tqdm
from typing import Any, Dict, Tuple, List, Optional
from collections.abc import Mapping

from crl.buffers import DictBuffer, ZBuffer, TrajectoryBuffer
from crl.envs import make_continual_episodic_env, make_env, get_task_sequence

print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)

# ==================================================
# Configs
# ==================================================


@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2


@dataclasses.dataclass
class ForwardArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class BackwardArchiConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True


@dataclasses.dataclass
class FBArchiConfig:
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig = dataclasses.field(default_factory=ForwardArchiConfig)
    b: BackwardArchiConfig = dataclasses.field(default_factory=BackwardArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)


@dataclasses.dataclass
class FBModelConfig:
    obs_dim: int = -1
    action_dim: int = -1
    device: str = "cpu"
    archi: FBArchiConfig = dataclasses.field(default_factory=FBArchiConfig)
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    norm_obs: bool = True


@dataclasses.dataclass
class CriticArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class DiscriminatorArchiConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 2


@dataclasses.dataclass
class FBcprArchiConfig(FBArchiConfig):
    critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)
    discriminator: DiscriminatorArchiConfig = dataclasses.field(
        default_factory=DiscriminatorArchiConfig
    )


@dataclasses.dataclass
class FBcprModelConfig(FBModelConfig):
    archi: FBcprArchiConfig = dataclasses.field(default_factory=FBcprArchiConfig)


@dataclasses.dataclass
class FBTrainConfig:
    lr_f: float = 1e-4
    lr_b: float = 1e-4
    lr_actor: float = 1e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 1.0
    train_goal_ratio: float = 0.5
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float | None = 0.99
    use_mix_rollout: bool = False
    update_z_every_step: int = 150
    z_buffer_size: int = 10000


@dataclasses.dataclass
class FBcprTrainConfig(FBTrainConfig):
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    # the z distribution for rollouts (when agent.use_mix_rollout=1) and for the mini-batches used
    # in the network updates is:
    # - a fraction of 'expert_asm_ratio' zs from expert trajectory encoding
    # - a fraction of 'train_goal_ratio' zs from goal encoding (goals sampled from the train buffer)
    # - the remaining fraction from the uniform distribution
    expert_asm_ratio: float = 0
    # a fraction of 'relabel_ratio' transitions in each mini-batch are relabeled with
    # a z sampled from the above distribution
    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0


@dataclasses.dataclass
class EnvConfig:
    domain_name: str = "dmcontrol"
    task_list: List[str] = dataclasses.field(default_factory=lambda: ["walker/stand", "walker/walk", "walker/run"])
    episode_len: int = 1000
    task_switch_prob: float = .01  # Probability of switching task at each episode reset (1.0 = always switch)


@dataclasses.dataclass
class Config:
    model: FBcprModelConfig = dataclasses.field(default_factory=FBcprModelConfig)
    train: FBcprTrainConfig = dataclasses.field(default_factory=FBcprTrainConfig)
    cudagraphs: bool = False
    compile: bool = False
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    seed: int = 0
    num_episodes: int = 10**4
    buffer_size: int = 3 * 10**3

    # Exploration configuration for epistemically-guided exploration (FBEE)
    exploration_update_freq: int = 5
    exploration_num_z_samples: int = 30  # Number of candidate z's to evaluate for uncertainty-based exploration
    exploration_num_obs: int = 1000  # Number of observations to sample and aggregate uncertainty over
    exploration_f_uncertainty: bool = False  # If True, use F-uncertainty (trace of cov); else use Q-uncertainty (std)
    # eval
    eval_freq: int = 1000  # episodes
    train_freq: int = 2  # episodes
    save_freq: int = 2*10**4
    use_wandb: bool = True
    exp_name: str = "bfm"
    num_inference_samples: int = 10_0
    num_eval_episodes: int = 1


# ==================================================
# Helpers
# ==================================================


def load_model(path: str, device: str | None, cls: Any):
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device
    loaded_agent = cls(**loaded_config)
    safetensors.torch.load_model(
        loaded_agent, model_dir / "model.safetensors", device=device
    )
    # loaded_agent.load_state_dict(
    #     torch.load(model_dir / "model.pt", weights_only=True, map_location=device)
    # )
    return loaded_agent


def dict_to_config(source: Mapping, target: Any):
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])
        else:
            print(f"[WARNING] field {field} not found in source config")


def config_from_dict(source: Dict, config_class: Any):
    target = config_class()
    dict_to_config(source, target)
    return target


# ==================================================
# Neural Networks
# ==================================================

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: Any, target_net_params: Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


def soft_update_params(net, target_net, tau) -> None:
    tau = float(min(max(tau, 0), 1))
    net_params = tuple(x.data for x in net.parameters())
    target_net_params = tuple(x.data for x in target_net.parameters())
    _soft_update_params(net_params, target_net_params, tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


def build_backward(obs_dim, z_dim, cfg):
    return BackwardMap(obs_dim, z_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.norm)


def build_forward(obs_dim, z_dim, action_dim, cfg, output_dim=None):
    if cfg.ensemble_mode == "seq":
        return SequetialFMap(obs_dim, z_dim, action_dim, cfg)
    elif cfg.ensemble_mode == "vmap":
        raise NotImplementedError("vmap ensemble mode is currently not supported")

    assert (
        cfg.ensemble_mode == "batch"
    ), "Invalid value for ensemble_mode. Use {'batch', 'seq', 'vmap'}"
    return _build_batch_forward(obs_dim, z_dim, action_dim, cfg, output_dim)


def _build_batch_forward(
    obs_dim, z_dim, action_dim, cfg, output_dim=None, parallel=True
):
    if cfg.model == "residual":
        forward_cls = ResidualForwardMap
    elif cfg.model == "simple":
        forward_cls = ForwardMap
    else:
        raise ValueError(f"Unsupported forward_map model {cfg.model}")
    num_parallel = cfg.num_parallel if parallel else 1
    return forward_cls(
        obs_dim,
        z_dim,
        action_dim,
        cfg.hidden_dim,
        cfg.hidden_layers,
        cfg.embedding_layers,
        num_parallel,
        output_dim,
    )


def build_actor(obs_dim, z_dim, action_dim, cfg):
    if cfg.model == "residual":
        actor_cls = ResidualActor
    elif cfg.model == "simple":
        actor_cls = Actor
    else:
        raise ValueError(f"Unsupported actor model {cfg.model}")
    return actor_cls(
        obs_dim,
        z_dim,
        action_dim,
        cfg.hidden_dim,
        cfg.hidden_layers,
        cfg.embedding_layers,
    )


def build_discriminator(obs_dim, z_dim, cfg):
    return Discriminator(obs_dim, z_dim, cfg.hidden_dim, cfg.hidden_layers)


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardMap(nn.Module):
    def __init__(
        self, goal_dim, z_dim, hidden_dim, hidden_layers: int = 2, norm=True
    ) -> None:
        super().__init__()
        seq = [nn.Linear(goal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, z_dim)]
        if norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [
        linear(input_dim, hidden_dim, num_parallel),
        layernorm(hidden_dim, num_parallel),
        nn.Tanh(),
    ]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)


class ForwardMap(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        hidden_dim,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
        num_parallel: int = 2,
        output_dim=None,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        self.embed_z = simple_embedding(
            obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel
        )
        self.embed_sa = simple_embedding(
            obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel
        )

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
        seq += [linear(hidden_dim, output_dim if output_dim else z_dim, num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(
            torch.cat([obs, z], dim=-1)
        )  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(
            torch.cat([obs, action], dim=-1)
        )  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class SequetialFMap(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, cfg, output_dim=None):
        super().__init__()
        self.models = nn.ModuleList(
            [
                _build_batch_forward(
                    obs_dim, z_dim, action_dim, cfg, output_dim, parallel=False
                )
                for _ in range(cfg.num_parallel)
            ]
        )

    def forward(
        self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        predictions = [model(obs, z, action) for model in self.models]
        return torch.stack(predictions)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        hidden_dim,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
    ) -> None:
        super().__init__()

        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)

        seq = []
        for _ in range(hidden_layers):
            seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [linear(hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


class Discriminator(nn.Module):
    def __init__(self, obs_dim, z_dim, hidden_dim, hidden_layers) -> None:
        super().__init__()
        seq = [
            nn.Linear(obs_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        ]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)

    def compute_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, obs], dim=1)
        logits = self.trunk(x)
        return logits

    def compute_reward(
        self, obs: torch.Tensor, z: torch.Tensor, eps: float = 1e-7
    ) -> torch.Tensor:
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward


##########################
# Residual models
##########################


class ResidualBlock(nn.Module):
    def __init__(self, dim, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(dim, num_parallel)
        lin = linear(dim, dim, num_parallel)
        self.mlp = nn.Sequential(ln, lin, nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(input_dim, num_parallel)
        lin = linear(input_dim, output_dim, num_parallel)
        seq = [ln, lin] + ([nn.Mish()] if activation else [])
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)


class ResidualForwardMap(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        hidden_dim,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
        num_parallel: int = 2,
        output_dim=None,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.hidden_dim = hidden_dim

        self.embed_z = residual_embedding(
            obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel
        )
        self.embed_sa = residual_embedding(
            obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel
        )

        seq = [ResidualBlock(hidden_dim, num_parallel) for _ in range(hidden_layers)]
        seq += [
            Block(hidden_dim, output_dim if output_dim else z_dim, False, num_parallel)
        ]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(
            torch.cat([obs, z], dim=-1)
        )  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(
            torch.cat([obs, action], dim=-1)
        )  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class ResidualActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        hidden_dim,
        hidden_layers: int = 1,
        embedding_layers: int = 2,
    ) -> None:
        super().__init__()

        self.embed_z = residual_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = residual_embedding(obs_dim, hidden_dim, embedding_layers)

        seq = [ResidualBlock(hidden_dim) for _ in range(hidden_layers)] + [
            Block(hidden_dim, action_dim, False)
        ]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_parallel: int,
        bias: bool = True,
        device=None,
        dtype=None,
        reset_params=True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DenseParallel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        if n_parallel is None or (n_parallel == 1):
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
        else:
            self.weight = nn.Parameter(
                torch.empty((n_parallel, in_features, out_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty((n_parallel, 1, out_features), **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
            if self.bias is None:
                raise NotImplementedError
        if reset_params:
            self.reset_parameters()

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            weight_list = [m.weight.T for m in module_list]
            target_weight = torch.stack(weight_list, dim=0)
            self.weight.data.copy_(target_weight.data)
            if self.bias:
                bias_list = [ln.bias.unsqueeze(0) for ln in module_list]
                target_bias = torch.stack(bias_list, dim=0)
                self.bias.data.copy_(target_bias.data)

    # TODO why do these layers have their own reset scheme?
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.n_parallel is None or (self.n_parallel == 1):
            return F.linear(input, self.weight, self.bias)
        else:
            return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, n_parallel={}, bias={}".format(
            self.in_features, self.out_features, self.n_parallel, self.bias is not None
        )


class ParallelLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        n_parallel,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(ParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [
                normalized_shape,
            ]
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            if n_parallel is None or (n_parallel == 1):
                self.weight = nn.Parameter(
                    torch.empty([*self.normalized_shape], **factory_kwargs)
                )
                self.bias = nn.Parameter(
                    torch.empty([*self.normalized_shape], **factory_kwargs)
                )
            else:
                self.weight = nn.Parameter(
                    torch.empty(
                        [n_parallel, 1, *self.normalized_shape], **factory_kwargs
                    )
                )
                self.bias = nn.Parameter(
                    torch.empty(
                        [n_parallel, 1, *self.normalized_shape], **factory_kwargs
                    )
                )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def load_module_list_weights(self, module_list) -> None:
        with torch.no_grad():
            assert len(module_list) == self.n_parallel
            if self.elementwise_affine:
                ln_weights = [ln.weight.unsqueeze(0) for ln in module_list]
                ln_biases = [ln.bias.unsqueeze(0) for ln in module_list]
                target_ln_weights = torch.stack(ln_weights, dim=0)
                target_ln_bias = torch.stack(ln_biases, dim=0)
                self.weight.data.copy_(target_ln_weights.data)
                self.bias.data.copy_(target_ln_bias.data)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if self.elementwise_affine:
            return (norm_input * self.weight) + self.bias
        else:
            return norm_input

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps
        self.noise_upper_limit = high - self.loc
        self.noise_lower_limit = low - self.loc

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class FBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, FBModelConfig)
        obs_dim, action_dim = self.cfg.obs_dim, self.cfg.action_dim
        arch = self.cfg.archi

        # create networks
        self._backward_map = build_backward(obs_dim, arch.z_dim, arch.b)
        self._forward_map = build_forward(obs_dim, arch.z_dim, action_dim, arch.f)
        self._actor = build_actor(obs_dim, arch.z_dim, action_dim, arch.actor)
        self._obs_normalizer = (
            nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01)
            if self.cfg.norm_obs
            else nn.Identity()
        )

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_model(path, device, cls=cls)

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        safetensors_save_model(self, output_folder / "model.safetensors")
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn(
            (size, self.cfg.archi.z_dim), dtype=torch.float32, device=device
        )
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(
        self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True
    ) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    def reward_inference(
        self,
        next_obs: torch.Tensor,
        reward: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs.shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = (
                i * self.cfg.inference_batch_size,
                (i + 1) * self.cfg.inference_batch_size,
            )
            B = self.backward_map(next_obs[start_idx:end_idx].to(self.cfg.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(
        self, next_obs: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)


class FBcprModel(FBModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config_from_dict(kwargs, FBcprModelConfig)
        self._discriminator = build_discriminator(
            self.cfg.obs_dim, self.cfg.archi.z_dim, self.cfg.archi.discriminator
        )
        self._critic = build_forward(
            self.cfg.obs_dim,
            self.cfg.archi.z_dim,
            self.cfg.action_dim,
            self.cfg.archi.critic,
            output_dim=1,
        )

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        super()._prepare_for_train()
        self._target_critic = copy.deepcopy(self._critic)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor, z: torch.Tensor):
        return self._discriminator(self._normalize(obs), z)


class FBAgent:
    def __init__(self, **kwargs):
        self.cfg = config_from_dict(kwargs, Config)
        self.cfg.train.fb_target_tau = float(
            min(max(self.cfg.train.fb_target_tau, 0), 1)
        )
        self._model = FBModel(**dataclasses.asdict(self.cfg.model))
        self.setup_training()
        self.setup_compile()
        self._model.to(self.cfg.model.device)

    @property
    def device(self):
        return self._model.cfg.device

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = torch.optim.Adam(
            self._model._backward_map.parameters(),
            lr=self.cfg.train.lr_b,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = torch.optim.Adam(
            self._model._forward_map.parameters(),
            lr=self.cfg.train.lr_f,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = torch.optim.Adam(
            self._model._actor.parameters(),
            lr=self.cfg.train.lr_actor,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(
            x for x in self._model._forward_map.parameters()
        )
        self._target_forward_map_paramlist = tuple(
            x for x in self._model._target_forward_map.parameters()
        )
        self._backward_map_paramlist = tuple(
            x for x in self._model._backward_map.parameters()
        )
        self._target_backward_map_paramlist = tuple(
            x for x in self._model._target_backward_map.parameters()
        )

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(
            self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device
        )
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(
            self.cfg.train.z_buffer_size,
            self.cfg.model.archi.z_dim,
            self.cfg.model.device,
        )

    def setup_compile(self):
        print(f"compile {self.cfg.compile}")
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            print(f"compiling with mode '{mode}'")
            self.update_fb = torch.compile(
                self.update_fb, mode=mode
            )  # use fullgraph=True to debug for graph breaks
            self.update_actor = torch.compile(
                self.update_actor, mode=mode
            )  # use fullgraph=True to debug for graph breaks
            self.sample_mixed_z = torch.compile(
                self.sample_mixed_z, mode=mode, fullgraph=True
            )

        print(f"cudagraphs {self.cfg.cudagraphs}")
        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_fb = CudaGraphModule(self.update_fb, warmup=5)
            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

    def act(
        self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True
    ) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor | None = None, *args, **kwargs):
        # samples a batch from the z distribution used to update the networks
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)

        if train_goal is not None:
            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            goals = self._model._backward_map(train_goal[perm])
            goals = self._model.project_z(goals)
            mask = (
                torch.rand((self.cfg.train.batch_size, 1), device=self.device)
                < self.cfg.train.train_goal_ratio
            )
            z = torch.where(mask, goals, z)
        return z

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        obs, action, next_obs, terminated = (
            batch["observation"],
            batch["action"],
            batch["next"]["observation"],
            batch["next"]["terminated"],
        )
        discount = self.cfg.train.discount * ~terminated

        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs, next_obs = self._model._obs_normalizer(
                obs
            ), self._model._obs_normalizer(next_obs)

        torch.compiler.cudagraph_mark_step_begin()
        z = self.sample_mixed_z(train_goal=next_obs).clone()
        self.z_buffer.add(z)

        q_loss_coef = (
            self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        )
        clip_grad_norm = (
            self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        )

        torch.compiler.cudagraph_mark_step_begin()
        metrics = self.update_fb(
            obs=obs,
            action=action,
            discount=discount,
            next_obs=next_obs,
            goal=next_obs,
            z=z,
            q_loss_coef=q_loss_coef,
            clip_grad_norm=clip_grad_norm,
        )
        metrics.update(
            self.update_actor(
                obs=obs,
                action=action,
                z=z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            target_Fs = self._model._target_forward_map(
                next_obs, z, next_action
            )  # num_parallel x batch x z_dim
            target_B = self._model._target_backward_map(goal)  # batch x z_dim
            target_Ms = torch.matmul(
                target_Fs, target_B.T
            )  # num_parallel x batch x batch
            _, _, target_M = self.get_targets_uncertainty(
                target_Ms, self.cfg.train.fb_pessimism_penalty
            )  # batch x batch

        # compute FB loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        B = self._model._backward_map(goal)  # batch x z_dim
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.train.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(
                    next_Qs, self.cfg.train.fb_pessimism_penalty
                )  # batch
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                inv_cov = torch.inverse(cov)  # z_dim x z_dim
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)  # batch
                target_Q = (
                    implicit_reward.detach() + discount.squeeze() * next_Q
                )  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model._forward_map.parameters(), clip_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self._model._backward_map.parameters(), clip_grad_norm
            )
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        return self.update_td3_actor(obs=obs, z=z, clip_grad_norm=clip_grad_norm)

    def update_td3_actor(
        self, obs: torch.Tensor, z: torch.Tensor, clip_grad_norm: float | None
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q = self.get_targets_uncertainty(
            Qs, self.cfg.train.actor_pessimism_penalty
        )  # batch
        actor_loss = -Q.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model._actor.parameters(), clip_grad_norm
            )
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "q": Q.mean().detach()}

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def maybe_update_rollout_context(
        self, z: torch.Tensor | None, step_count: torch.Tensor
    ) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self.cfg.model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self.cfg.model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.cfg.model.device))
        else:
            z = self._model.sample_z(step_count.shape[0], device=self.cfg.model.device)
        return z

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if device is not None:
            loaded_config["model"]["device"] = device
        agent = cls(**loaded_config)
        optimizers = torch.load(str(path / "optimizers.pth"), weights_only=True)
        agent.actor_optimizer.load_state_dict(optimizers["actor_optimizer"])
        agent.backward_optimizer.load_state_dict(optimizers["backward_optimizer"])
        agent.forward_optimizer.load_state_dict(optimizers["forward_optimizer"])

        safetensors.torch.load_model(
            agent._model, path / "model/model.safetensors", device=device
        )
        return agent

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)
        # save optimizer
        torch.save(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "backward_optimizer": self.backward_optimizer.state_dict(),
                "forward_optimizer": self.forward_optimizer.state_dict(),
            },
            output_folder / "optimizers.pth",
        )
        # save model
        model_folder = output_folder / "model"
        model_folder.mkdir(exist_ok=True)
        self._model.save(output_folder=str(model_folder))


class FBcprAgent(FBAgent):
    def __init__(self, **kwargs):
        # make sure batch size is a multiple of seq_length
        seq_length = kwargs["model"]["seq_length"]
        batch_size = kwargs["train"]["batch_size"]
        kwargs["train"]["batch_size"] = int(
            torch.ceil(torch.tensor([batch_size / seq_length])) * seq_length
        )
        del seq_length, batch_size

        self.cfg = config_from_dict(kwargs, Config)
        self._model = FBcprModel(**dataclasses.asdict(self.cfg.model))
        self._model.to(self.cfg.model.device)
        self.setup_training()
        self.setup_compile()

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(
            x for x in self._model._target_critic.parameters()
        )

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(
                self.update_discriminator, mode=mode
            )
            self.encode_expert = torch.compile(
                self.encode_expert, mode=mode, fullgraph=True
            )

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(
                self.update_discriminator, warmup=5
            )
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    @torch.no_grad()
    def sample_mixed_z(
        self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs
    ):
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        p_goal = self.cfg.train.train_goal_ratio
        p_expert_asm = self.cfg.train.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(
            prob, num_samples=self.cfg.train.batch_size, replacement=True
        ).reshape(-1, 1)

        # zs obtained by encoding train goals
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        goals = self._model._backward_map(train_goal[perm])
        goals = self._model.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # zs obtained by encoding expert trajectories
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor):
        # encode expert trajectories through B
        B_expert = self._model._backward_map(next_obs).detach()  # batch x d
        B_expert = B_expert.view(
            self.cfg.train.batch_size // self.cfg.model.seq_length,
            self.cfg.model.seq_length,
            B_expert.shape[-1],
        )  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(
            z_expert, self.cfg.model.seq_length, dim=0
        )  # batch x d
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            train_batch["observation"].to(self.device),
            train_batch["action"].to(self.device),
            train_batch["next"]["observation"].to(self.device),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(
            self.device
        )
        expert_obs, expert_next_obs = (
            expert_batch["observation"].to(self.device),
            expert_batch["next"]["observation"].to(self.device),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = (
            self.cfg.train.grad_penalty_discriminator
            if self.cfg.train.grad_penalty_discriminator > 0
            else None
        )
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(
            train_goal=train_next_obs, expert_encodings=expert_z
        ).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = (
                torch.rand((self.cfg.train.batch_size, 1), device=self.device)
                <= self.cfg.train.relabel_ratio
            )
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = (
            self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        )
        clip_grad_norm = (
            self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        )

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor,
        real_z: torch.Tensor,
        fake_obs: torch.Tensor,
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=real_obs.device)
        interpolates = torch.cat(
            [
                (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
                (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True),
            ],
            dim=1,
        )
        d_interpolates = self._model._discriminator.compute_logits(
            interpolates[:, 0: real_obs.shape[1]], interpolates[:, real_obs.shape[1]:]
        )
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor,
        expert_z: torch.Tensor,
        train_obs: torch.Tensor,
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        expert_logits = self._model._discriminator.compute_logits(
            obs=expert_obs, z=expert_z
        )
        unlabeled_logits = self._model._discriminator.compute_logits(
            obs=train_obs, z=train_z
        )
        # these are equivalent to binary cross entropy
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        if grad_penalty is not None:
            wgan_gp = self.gradient_penalty_wgan(
                expert_obs, expert_z, train_obs, train_z
            )
            loss += grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_parallel = self.cfg.model.archi.critic.num_parallel
        # compute target critic
        with torch.no_grad():
            reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_Qs = self._model._target_critic(
                next_obs, z, next_action
            )  # num_parallel x batch x 1
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(
                next_Qs, self.cfg.train.critic_pessimism_penalty
            )
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # compute critic loss
        Qs = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)

        # compute discriminator reward loss
        Qs_discriminator = self._model._critic(
            obs, z, action
        )  # num_parallel x batch x (1 or n_bins)
        _, _, Q_discriminator = self.get_targets_uncertainty(
            Qs_discriminator, self.cfg.train.actor_pessimism_penalty
        )  # batch

        # compute fb reward loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(
            Qs_fb, self.cfg.train.actor_pessimism_penalty
        )  # batch

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        actor_loss = (
            -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()
        )

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._model._actor.parameters(), clip_grad_norm
            )
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics


def eval(agent, config, t):
    task_zs = {}
    for task in config.env.task_list:
        # Create environment for this task
        eval_env = make_env(
            env_id=config.env.domain_name,
            task=task,
        )

        # Collect samples using uniform random policy
        action_space = eval_env.action_space
        next_obs_list, rewards_list = [], []

        print(f"Collecting {config.num_inference_samples} samples from {task} using uniform policy...")
        num_collected = 0

        obs, _ = eval_env.reset()
        terminated = truncated = False

        while num_collected < config.num_inference_samples:
            if terminated or truncated:
                obs, _ = eval_env.reset()

            # Sample uniform random action
            if hasattr(action_space, 'sample'):
                action = action_space.sample()
            else:
                # Fallback for Box spaces
                action = np.random.uniform(
                    action_space.low,
                    action_space.high,
                    size=action_space.shape
                ).astype(action_space.dtype)

            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            next_obs_list.append(next_obs)
            rewards_list.append(reward)
            num_collected += 1

            obs = next_obs

        # Infer z from collected samples
        next_obs = torch.from_numpy(
            np.stack(next_obs_list).astype(np.float32)
        ).to(agent.device)
        rewards = torch.from_numpy(
            np.array(rewards_list, dtype=np.float32).reshape(-1, 1)
        ).to(agent.device)

        with torch.no_grad(), eval_mode(agent._model):
            z = agent._model.reward_inference(next_obs=next_obs, reward=rewards)

        task_zs[task] = z.cpu().numpy().flatten()
        print(f"Inferred z for {task} from {len(rewards_list)} samples")

    # Compute pairwise cosine similarities between task z vectors
    tasks = list(task_zs.keys())
    for i, task1 in enumerate(tasks):
        for task2 in tasks[i+1:]:
            z1 = task_zs[task1]
            z2 = task_zs[task2]
            cosine_sim = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
            if config.use_wandb:
                # Log with task names for easy identification in wandb
                wandb.log({
                    f"eval/z_similarity_{task1}_vs_{task2}": cosine_sim,
                }, step=t)

    # Now evaluate each task using the inferred z
    for task in config.env.task_list:
        z = torch.tensor(task_zs[task], device=agent.device).reshape(1, -1)
        eval_env = make_env(
            env_id=config.env.domain_name,
            task=task,
        )
        num_ep = config.num_eval_episodes
        total_reward = np.zeros((num_ep,), dtype=np.float32)
        for ep in range(num_ep):
            obs, _ = eval_env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                with torch.no_grad(), eval_mode(agent._model):
                    obs_tensor = torch.tensor(
                        obs,
                        device=agent.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    action = agent.act(obs=obs_tensor, z=z, mean=True).cpu().numpy()[0]
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward[ep] += reward
                obs = next_obs
        metrics = {
            "reward": np.mean(total_reward),
            "reward#std": np.std(total_reward),
        }
        if config.use_wandb:
            wandb.log({f"eval/{task}/{k}": v for k, v in metrics.items()}, step=t)

"""
Utility: select_exploratory_z

This function implements epistemically-guided exploration for forward-backward representations
as described in "Epistemically-guided forward-backward exploration" (Armengol Urpí et al., 2025).
Paper: https://arxiv.org/pdf/2507.05477
Original code: https://github.com/nuria95/fbee/blob/main/url_benchmark/agent/fb_ddpg.py

The function selects an exploratory latent goal z by maximizing epistemic uncertainty:
- If sampling=True:
  * Samples `num_z_samples` candidate z's from the prior
  * Computes actions for each z via the agent's actor
  * Queries the ensemble forward_map to obtain ensemble predictions
  * Computes an epistemic-uncertainty score per candidate z:
    - If f_uncertainty=True: trace-of-covariance of forward outputs F
    - If f_uncertainty=False: std across ensemble of Q-values (Q = F · z)
  * Returns the z with the highest epistemic score

- If sampling=False: falls back to sampling a single random z

The function is adapted to work directly with FBcprAgent, using:
- agent._model.forward_map() for ensemble predictions (returns num_parallel x batch x z_dim)
- agent._model.actor() for action computation (returns distribution with .mean)
- agent._model.sample_z() for z sampling

Example cfg:
cfg = {
    "sampling": True,
    "num_z_samples": 100,
    "f_uncertainty": False,  # Use Q-uncertainty (std of Q-values) instead of F-uncertainty
}
"""



def select_exploratory_z(
    obs: np.ndarray,
    agent: FBcprAgent,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Select an exploratory latent goal z using ensemble disagreement (FBEE method).

    Based on: https://arxiv.org/pdf/2507.05477
    Adapted from: https://github.com/nuria95/fbee/blob/main/url_benchmark/agent/fb_ddpg.py

    Args:
        obs: observation(s) - can be single observation (1D numpy array) or batch (N x obs_dim).
             If single observation, uncertainty is computed for that observation.
             If batch, uncertainty is aggregated (mean) across observations for each candidate z.
        agent: FBcprAgent instance with forward_map ensemble and actor
        cfg: dictionary with keys:
            - sampling (bool): if True, sample multiple z candidates and select by uncertainty
            - num_z_samples (int): number of candidate z's to sample when sampling=True
            - f_uncertainty (bool): if True, use trace-of-covariance of F; else use Q-value std
            - norm_z (bool): whether to normalize z (usually True, handled by agent._model.project_z)
        device: 'cuda' or 'cpu' (optional, defaults to agent.device)
        std_for_actor: std for actor when computing actions (optional, defaults to agent._model.cfg.actor_std)

    Returns:
        selected_z: numpy array shape (1, z_dim) containing the chosen exploratory z.
    """

    # Convert obs to tensor and place on device. Handle both single obs and batch.
    if isinstance(obs, np.ndarray):
        obs_t = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)
    elif isinstance(obs, torch.Tensor):
        obs_t = obs.to(device=device, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)
    else:
        raise TypeError("obs must be numpy array or torch.Tensor")

    num_obs = obs_t.shape[0]  # Number of observations to aggregate over

    # Sample candidate z's
    z_all = agent._model.sample_z(config.exploration_num_z_samples, device=agent.device)  # config.exploration_num_z_samples x z_dim
    z_all = z_all.to(dtype=torch.float32)

    # Process candidates in batches to avoid OOM
    batch_size = min(32, config.exploration_num_z_samples)  # Process in smaller batches
    epistemic_scores_list = []

    with torch.no_grad():
        for batch_start in range(0, config.exploration_num_z_samples, batch_size):
            batch_end = min(batch_start + batch_size, config.exploration_num_z_samples)
            z_batch = z_all[batch_start:batch_end]  # batch_size x z_dim
            num_batch = z_batch.shape[0]

            # Expand obs and z to compute uncertainty for all (obs, z) pairs
            # obs_t: (num_obs, obs_dim), z_batch: (num_batch, z_dim)
            # We want to evaluate each z with each observation
            obs_expanded = obs_t.unsqueeze(1).expand(-1, num_batch, -1)  # (num_obs, num_batch, obs_dim)
            z_expanded = z_batch.unsqueeze(0).expand(num_obs, -1, -1)  # (num_obs, num_batch, z_dim)

            # Flatten to process all pairs together
            obs_flat = obs_expanded.reshape(-1, obs_t.shape[1])  # (num_obs * num_batch, obs_dim)
            z_flat = z_expanded.reshape(-1, z_batch.shape[1])  # (num_obs * num_batch, z_dim)

            # Compute deterministic action means for each (obs, z) pair
            dist = agent._model.actor(obs_flat, z_flat, agent._model.cfg.actor_std)
            acts = dist.mean.to(dtype=torch.float32)  # (num_obs * num_batch, action_dim)

            # Query forward_map: returns shape (num_parallel, num_obs * num_batch, z_dim)
            F1_flat = agent._model.forward_map(obs_flat, z_flat, acts)  # num_parallel x (num_obs * num_batch) x z_dim

            # Ensure F1 has ensemble dimension
            if F1_flat.dim() == 3:
                # Reshape back to separate obs and z dimensions
                F1_reshaped = F1_flat.reshape(F1_flat.shape[0], num_obs, num_batch, F1_flat.shape[2])  # num_parallel x num_obs x num_batch x z_dim

                # Compute epistemic scores for each candidate z, aggregated across observations
                z_scores = []  # Will store scores for each z candidate
                for z_idx in range(num_batch):
                    # Get F predictions for this z across all observations: (num_parallel, num_obs, z_dim)
                    F1_z = F1_reshaped[:, :, z_idx, :]

                    if config.exploration_f_uncertainty:
                        # Compute trace-of-covariance for each observation, then aggregate
                        traces_per_obs = []
                        for obs_idx in range(num_obs):
                            mat = F1_z[:, obs_idx, :].transpose(0, 1)  # z_dim x num_parallel
                            cov = torch.cov(mat)
                            traces_per_obs.append(torch.trace(cov).item())
                        # Aggregate: mean uncertainty across observations for this z
                        z_scores.append(np.mean(traces_per_obs))
                    else:
                        # Compute Q-values: Q = sum(F * z, dim=-1) per ensemble member
                        z_candidate = z_batch[z_idx]  # (z_dim,)
                        # F1_z: (num_parallel, num_obs, z_dim), z_candidate: (z_dim,)
                        Q1_z = torch.einsum('noz, z -> no', F1_z, z_candidate)  # num_parallel x num_obs
                        # Aggregate: mean std across observations for this z
                        std_per_obs = Q1_z.std(dim=0)  # (num_obs,)
                        z_scores.append(std_per_obs.mean().item())

                epistemic_scores_list.append(torch.as_tensor(z_scores, device=agent.device, dtype=torch.float32))
            else:
                # Unexpected shape: use zero scores for this batch
                epistemic_scores_list.append(torch.zeros(num_batch, device=agent.device, dtype=torch.float32))

    # Concatenate all batch scores
    epistemic_scores = torch.cat(epistemic_scores_list)  # num_zs

    # Choose argmax
    idx = int(torch.argmax(epistemic_scores).cpu().item())
    selected_z = z_all[idx].cpu().numpy()  # shape: (z_dim,)
    # Return with batch dimension to match sample_z(1, ...) which returns (1, z_dim)
    return selected_z[np.newaxis, :]  # shape: (1, z_dim)


if __name__ == "__main__":

    config = tyro.cli(Config)

    if config.use_wandb:
        wandb.init(
            project="continual-rl",
            name=f"{config.exp_name}-{config.env.domain_name}-OnlineFBcpr",
            config=dataclasses.asdict(config)
        )

    env = make_continual_episodic_env(
        env_id=config.env.domain_name,
        task_list=config.env.task_list,
        episode_length=config.env.episode_len,
        task_switch_prob=config.env.task_switch_prob,
        seed=config.seed,
    )

    # Extract observation and action dimensions from environment
    obs_space = env.observation_space
    action_space = env.action_space

    # Get observation dimension
    if hasattr(obs_space, 'shape'):
        if len(obs_space.shape) == 0:
            obs_dim = 1
        else:
            obs_dim = int(np.prod(obs_space.shape))
    else:
        # Fallback: reset and check observation
        obs, _ = env.reset()
        obs_dim = int(np.prod(obs.shape))

    # Get action dimension
    if hasattr(action_space, 'shape'):
        if len(action_space.shape) == 0:
            action_dim = 1
        else:
            action_dim = int(np.prod(action_space.shape))
    elif hasattr(action_space, 'n'):
        # Discrete action space
        action_dim = int(action_space.n)
    else:
        # Fallback: sample an action
        action = action_space.sample()
        action_dim = int(np.prod(action.shape) if hasattr(action, 'shape') else 1)

    # Set dimensions in config
    config.model.obs_dim = obs_dim
    config.model.action_dim = action_dim

    agent = FBcprAgent(**dataclasses.asdict(config))

    z = agent._model.sample_z(1, device=agent.device)  # Initialize z

    replay_buffer: Dict[str, Any] = {
        "train": DictBuffer(capacity=config.buffer_size, device=agent.device),
        "expert_slicer": TrajectoryBuffer(
            capacity=config.buffer_size,
            device=agent.device,
            seq_length=config.model.seq_length,
        ),
    }

    # Store last window_size episode (next_obs, reward) and reward
    # for quick z inference and to compute regret
    window_size: int = 5  # number of last episodes to store.
    nextobs_buffer: List[torch.Tensor] = []
    rewards_buffer: List[float] = []

    # Create task mapping for visualization (task name -> integer ID)
    task_list = config.env.task_list
    task_to_id = {task: idx for idx, task in enumerate(task_list)}

    for episode in tqdm(range(config.num_episodes)):

        # --------------------------------------------------
        # Exploration (choosing z to act in this episode)
        # --------------------------------------------------
        if len(nextobs_buffer) > config.exploration_num_obs:
            if  episode % config.exploration_update_freq == 0:
                # Use nextobs_buffer (observations from current task) for task-specific exploration
                # This focuses exploration on the current task being solved
                with torch.no_grad(), eval_mode(agent._model):
                    # Use observations from nextobs_buffer (current task)
                    # Take the most recent observations
                    obs_tensors = nextobs_buffer[-config.exploration_num_obs:]
                    exploration_obs = torch.stack(obs_tensors).cpu().to(dtype=torch.float32).numpy()  # (num_obs, obs_dim)

                    z_exploration = select_exploratory_z(
                        obs=exploration_obs,
                        agent=agent,
                        cfg=config,
                    )
                    z = torch.as_tensor(z_exploration, device=agent.device, dtype=torch.float32)
            else:  # Exploitation
                with torch.no_grad(), eval_mode(agent._model):
                    next_obs_tensor = torch.stack(nextobs_buffer).to(agent.device)
                    reward_tensor = torch.tensor(
                        rewards_buffer, dtype=torch.float32, device=agent.device
                    ).reshape(-1, 1)
                    z = agent._model.reward_inference(next_obs=next_obs_tensor, reward=reward_tensor)
        else:
            # Fallback to random exploration if not enough data
            z = agent._model.sample_z(1, device=agent.device)

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------
        if episode % config.eval_freq == 0:
            eval(agent, config, episode)

        # --------------------------------------------------
        # Train the agent
        # --------------------------------------------------
        if episode % config.train_freq == 0 and len(replay_buffer["train"]) > config.train.batch_size and episode > 10:
            metrics = agent.update(replay_buffer, episode)

            if config.use_wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=episode)

        # --------------------------------------------------
        # Online env interaction
        # --------------------------------------------------
        obs, info = env.reset()
        # Extract current task for logging
        current_task = info.get('task', 'unknown') if info else 'unknown'
        current_task_id = task_to_id.get(current_task, -1)

        terminated = truncated = False
        episode_reward = 0.0
        # Collect episode data
        episode_transitions = []

        while not (terminated or truncated):
            action_tensor = agent.act(
                torch.as_tensor(obs, device=agent.device, dtype=torch.float32).unsqueeze(0),
                z,
                mean=False
            )
            action = action_tensor.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            # TODO: probably nextobs_buffer should contains only last non-exploratory data?
            nextobs_buffer.append(torch.from_numpy(next_obs).float())
            rewards_buffer.append(reward)

            # Ensure z has shape (1, z_dim) - it might be (z_dim,) from reward_inference
            z_np = z.cpu().numpy().astype(np.float32)
            if z_np.ndim == 1:
                z_np = z_np[np.newaxis, :]  # Add batch dimension: (z_dim,) -> (1, z_dim)
            # z_np should now be (1, z_dim)

            # Ensure all arrays have shape (1, feature_dim) for consistent buffer handling
            # This prevents shape mismatches when the buffer wraps around
            def ensure_2d(arr, is_scalar=False):
                """Ensure array has shape (1, feature_dim)"""
                arr = np.asarray(arr)
                if is_scalar or arr.ndim == 0:
                    # Scalar: reshape to (1, 1)
                    return arr.reshape(1, 1)
                elif arr.ndim == 1:
                    # 1D array: add batch dimension (1, feature_dim)
                    return arr[np.newaxis, :]
                else:
                    # Already 2D or more: should be (1, feature_dim)
                    return arr

            data = {
                "observation": ensure_2d(obs),
                "action": ensure_2d(action),
                "z": z_np,  # Already (1, z_dim)
                "next": {
                    "observation": ensure_2d(next_obs),
                    "terminated": ensure_2d(terminated or truncated, is_scalar=True),
                },
                "reward": ensure_2d(reward, is_scalar=True),
            }
            replay_buffer["train"].extend(data)
            episode_transitions.append(data)

            obs = next_obs

        # Store episode data for expert buffer (full episode as trajectory)
        # TrajectoryBuffer expects a single dict per episode with concatenated timesteps
        # Concatenate all transitions into a single episode dictionary
        episode_dict = {
            "observation": np.concatenate([t["observation"] for t in episode_transitions], axis=0),
            "action": np.concatenate([t["action"] for t in episode_transitions], axis=0),
            "z": np.concatenate([t["z"] for t in episode_transitions], axis=0),
            "reward": np.concatenate([t["reward"] for t in episode_transitions], axis=0),
            "next": {
                "observation": np.concatenate([t["next"]["observation"] for t in episode_transitions], axis=0),
                "terminated": np.concatenate([t["next"]["terminated"] for t in episode_transitions], axis=0),
            },
        }
        # Only store if episode had positive rewards (simple heuristic)
        # TODO: fix this, better to keep only trajectories with very high reward
        if (episode_reward > 0 and len(episode_transitions) > 0) or episode < 10:
            replay_buffer["expert_slicer"].extend([episode_dict])

        # --------------------------------------------------
        # Online metrics
        # --------------------------------------------------
        if config.use_wandb:
            wandb.log({
                "metrics/reward_per_episode": episode_reward,
                "metrics/task_id": current_task_id,  # Numeric ID for step plot (different colors per task)
                "metrics/task_name": current_task,   # Task name for reference
            }, step=episode)

        if episode % window_size == 0:
            # After window_size episode clear the buffers
            if config.use_wandb:
                wandb.log({
                    "debug/len_nextobs_buffer": len(nextobs_buffer),
                    "debug/len_episode": len(episode_dict["observation"])
                }, step=episode)
            nextobs_buffer.clear()
            rewards_buffer.clear()

