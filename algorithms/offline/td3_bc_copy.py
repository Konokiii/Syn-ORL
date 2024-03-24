# source: https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/2106.06860.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from icl4rl.state_action_annotations import *
from icl4rl.add_annotation_copy import *

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # Domain
    source_domain: str = "halfcheetah"
    target_domain: str = 'walker2d'
    source_dataset: str = 'medium-replay'
    target_dataset: str = 'medium-replay'

    # Language embedding
    encoding_only: bool = False
    enc_batch_size: int = 64
    enable_language_encoding: bool = True
    pretrained_LM: str = 'sentence-transformers/all-mpnet-base-v2'
    emb_mode: str = 'avg'  # Choose from 'avg', 'cls'
    prefix_name: str = 'none'
    suffix_name: str = 'none'
    normalize_embedding: bool = False

    # Cross domain setup
    data_ratio: float = 1.0
    cross_training_mode: str = 'ZeroShot'  # Choose from 'ZeroShot', 'SymCoT', 'None'

    # Model arch
    hidden_arch: str = '256-256'  # For example, this means two hidden layer of size 256

    # Experiment
    device: str = "cuda"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount ffor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "ICL4RL"
    group: str = "TD3_BC-D4RL"
    name: str = "TD3_BC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.source_domain}-{self.target_domain}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, enable_language_encoding: bool,
        domain: str, tokenizer, language_model, prefix_name, suffix_name, emb_mean, emb_std, emb_mode
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            if enable_language_encoding:
                prefix = ALL_ANNOTATIONS_DICT[prefix_name]['state']
                suffix = ALL_ANNOTATIONS_DICT[suffix_name]['state']
                state = encode(state, domain, tokenizer, language_model, prefix, suffix, emb_mode, device)
                if emb_mean is not None:
                    state = normalize_states(state, emb_mean.cpu().numpy(), emb_std.cpu().numpy())
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, arch: str = '256-256'):
        super(Actor, self).__init__()
        in_dim = state_dim
        module_list = []
        for i, hidden_size in enumerate(arch.split('-')):
            hidden_size = int(hidden_size)
            module_list.append(nn.Linear(in_dim, hidden_size))
            module_list.append(nn.ReLU())
            in_dim = hidden_size
        module_list.append(nn.Linear(in_dim, action_dim))
        module_list.append(nn.Tanh())
        self.net = nn.Sequential(*module_list)

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, action_dim),
        #     nn.Tanh(),
        # )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, arch: str = '256-256'):
        super(Critic, self).__init__()

        in_dim = state_dim + action_dim
        module_list = []
        for i, hidden_size in enumerate(arch.split('-')):
            hidden_size = int(hidden_size)
            module_list.append(nn.Linear(in_dim, hidden_size))
            module_list.append(nn.ReLU())
            in_dim = hidden_size
        module_list.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*module_list)

        # self.net = nn.Sequential(
        #     nn.Linear(state_dim + action_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1),
        # )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            alpha: float = 2.5,
            device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


def preprocess_states_and_actions(dataset, env_name: str, env, normalize_reward: bool = False, normalize: bool = True):
    if normalize_reward:
        modify_reward(dataset, env_name)

    if normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    return wrap_env(env, state_mean=state_mean, state_std=state_std)


# TODO: Should input nparray
def normalize_embedding(buffer: ReplayBufferProMax):
    state_mean, state_std = compute_mean_std(buffer._state_embeddings, eps=1e-3)
    buffer._state_embeddings = normalize_states(
        buffer._state_embeddings, state_mean, state_std
    )
    buffer._next_state_embeddings = normalize_states(
        buffer._next_state_embeddings, state_mean, state_std
    )
    return state_mean, state_std


class Domain:
    def __init__(self,
                 domain_name: str,
                 data_type: str,
                 ):
        self.domain_name = domain_name
        self.data_type = data_type
        self.env_name = '%s-%s-v2' % (domain_name, data_type)
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.raw_dataset = None
        self.emb_dataset = None

    def load_d4rl_qlearning_dataset_with_ratio(self, data_ratio=1.0, seed=0):
        raw_dataset = d4rl.qlearning_dataset(self.env)
        n_transitions = raw_dataset['observations'].shape[0]
        if data_ratio != 1.0:
            np.random.seed(seed)
            sample_idx = np.random.randint(n_transitions, size=n_transitions*data_ratio)
    def reset_env_state_stats(self, state_mean, state_std):
        self.env = wrap_env(self.env, state_mean=state_mean, state_std=state_std)


# @pyrallis.wrap()
def run_TD3_BC(config: TrainConfig):
    # TODO: Write the following two blocks into the same class object. Note eval_actor and encode functions.
    # TODO: Need a proper place to normalize the state and action and the environment and test environment!!!
    source_domain = Domain(config.source_domain, config.source_dataset) if config.cross_training_mode != 'None' else None
    target_domain = Domain(config.target_domain, config.target_dataset)

    domain_list = [target_domain]
    if source_domain is not None:
        domain_list.append(source_domain)

    # Load d4rl data and language encoding
    for domain in domain_list:


    data ratio
    language_enc






    # source_env = preprocess_states_and_actions(source_raw_dataset, source_env_name, source_env, config.normalize_reward,
    #                                            config.normalize)
    # source_buffer = ReplayBufferProMax(
    #     source_state_dim,
    #     source_action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    # source_buffer.load_d4rl_dataset(source_raw_dataset)
    #
    #
    # target_raw_dataset = d4rl.qlearning_dataset(target_env)
    # target_env = preprocess_states_and_actions(target_raw_dataset, target_env_name, target_env, config.normalize_reward,
    #                                            config.normalize)
    # target_buffer = ReplayBufferProMax(
    #     target_state_dim,
    #     target_action_dim,
    #     config.buffer_size,
    #     config.device,
    # )
    # target_buffer.load_d4rl_dataset(target_raw_dataset)

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_LM)
    language_model = AutoModel.from_pretrained(config.pretrained_LM).to(config.device)
    language_model.eval()
    language_embedding_dim = language_model.config.hidden_size
    # TODO: (1)Find ways to support different annotations for state and action;
    #       (2)Find ways to support different annoations for source and target domain
    if config.max_timesteps < 100:  # This is only for debug
        source_buffer.retain_data_ratio(data_ratio=512 / source_buffer._size)
        target_buffer.retain_data_ratio(data_ratio=512 / target_buffer._size)

    source_buffer.encode_raw_d4rl_data(config.source_domain, config.source_dataset, tokenizer, language_model,
                                       config.prefix_name, config.suffix_name, batch_size=config.enc_batch_size,
                                       encoding_only=config.encoding_only, emb_mode=config.emb_mode)
    # TODO: Improve encoding_only case.
    if config.encoding_only:
        return

    target_buffer.encode_raw_d4rl_data(config.target_domain, config.target_dataset, tokenizer, language_model,
                                       config.prefix_name, config.suffix_name, batch_size=config.enc_batch_size,
                                       emb_mode=config.emb_mode)

    emb_mean, emb_std = None, None
    if config.normalize_embedding:
        normalize_embedding(source_buffer)
        # TODO: They are tensor...
        emb_mean, emb_std = normalize_embedding(target_buffer)

    target_buffer.retain_data_ratio(data_ratio=config.data_ratio)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, target_env)

    # TODO: Modify the action dimension and max action when having decoder.
    input_state_dim = language_embedding_dim if config.enable_language_encoding else target_state_dim
    hidden_arch = config.hidden_arch

    actor = Actor(input_state_dim, target_action_dim, target_max_action, arch=hidden_arch).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(input_state_dim, target_action_dim, arch=hidden_arch).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(input_state_dim, target_action_dim, arch=hidden_arch).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    # TODO: kwargs also contains max_action.
    kwargs = {
        "max_action": target_max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * target_max_action,
        "noise_clip": config.noise_clip * target_max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    print("---------------------------------------")
    print(
        f"Training TD3 + BC, Source_Domain: {config.source_domain}, Target_Domain: {config.target_domain}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = TD3_BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    # TODO: Implement 'finetune' mode; 'random co-training.
    evaluations = []
    for t in tqdm(range(int(config.max_timesteps)), desc='RL Agent Training'):
        # TODO: modify sample_action_embedding option.
        batch = []

        if config.cross_training_mode == 'SymCoT':
            source_batch = source_buffer.sample(config.batch_size // 2,
                                                sample_state_embedding=config.enable_language_encoding)
            target_batch = target_buffer.sample(config.batch_size // 2,
                                                sample_state_embedding=config.enable_language_encoding)
            batch = [torch.cat(i, dim=0) for i in zip(source_batch, target_batch)]
        elif config.cross_training_mode == 'RandomCoT':
            pass
        elif config.cross_training_mode == 'ZeroShot':
            batch = source_buffer.sample(config.batch_size, sample_state_embedding=config.enable_language_encoding)
        elif config.cross_training_mode == 'None':  # Target domain data training only
            batch = target_buffer.sample(config.batch_size, sample_state_embedding=config.enable_language_encoding)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                target_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                enable_language_encoding=config.enable_language_encoding,
                domain=config.target_domain,
                tokenizer=tokenizer,
                language_model=language_model,
                prefix_name=config.prefix_name,
                suffix_name=config.suffix_name,
                emb_mean=emb_mean,
                emb_std=emb_std,
                emb_mode=config.emb_mode
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = target_env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )

# if __name__ == "__main__":
#     train()
