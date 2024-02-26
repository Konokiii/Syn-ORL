import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

TensorBatch = List[torch.Tensor]


def add_annotation(raw_vectors: np.ndarray, domain_name: str, prefix_annotation_dict: Optional[Dict[str, list]] = None,
                   suffix_annotation_dict: Optional[Dict[str, list]] = None):
    if prefix_annotation_dict is not None and prefix_annotation_dict.get(domain_name, None) is None:
        raise KeyError('Unsupported domain prefix annotation.')
    if suffix_annotation_dict is not None and suffix_annotation_dict.get(domain_name, None) is None:
        raise KeyError('Unsupported domain suffix annotation.')
    # TODO: (1) Handle the case when either of the annotation_dict is None.
    #       For example, initialize them as ['' for _ in range]
    #       (2) Check whether the length of vector and annotation are the same.
    prefix = prefix_annotation_dict[domain_name]
    suffix = suffix_annotation_dict[domain_name]
    # Handle one-dim state during inference
    if raw_vectors.ndim == 1:
        raw_vectors = raw_vectors.reshape(1, -1)
    if not (len(prefix) == len(suffix) == raw_vectors.shape[1]):
        raise ValueError("The annotation lists and the dimension of raw vectors must have the same length.")
    prefix_list = [s + ': ' if s != '' else '' for s in prefix]
    suffix_list = [' ' + s + ',' if s != '' else ',' for s in suffix]
    raw_vectors = raw_vectors.round(decimals=5).astype('str')
    readable = np.char.add(np.char.add(prefix_list, raw_vectors), suffix_list)
    readable = np.apply_along_axis(' '.join, 1, readable).tolist()
    return readable


def LM_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@torch.no_grad()
def encode(raw_vectors: np.ndarray, domain_name: str, tokenizer, language_model,
           prefix_annotation_dict: Optional[Dict[str, list]] = None,
           suffix_annotation_dict: Optional[Dict[str, list]] = None, device: str = 'cpu'):
    readable = add_annotation(raw_vectors, domain_name, prefix_annotation_dict, suffix_annotation_dict)
    lm_inputs = tokenizer(readable, padding=True, truncation=True, return_tensors='pt').to(device)
    lm_outputs = language_model(**lm_inputs)
    embeddings = LM_mean_pooling(lm_outputs, lm_inputs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


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


class ReplayBufferProMax(ReplayBuffer):
    def __init__(self, state_dim: int,
                 action_dim: int,
                 buffer_size: int,
                 device: str = "cpu",
                 ):
        super().__init__(state_dim, action_dim, buffer_size, device)
        # self._buffer_size = buffer_size
        # self._pointer = 0
        # self._size = 0
        #
        # self._states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._actions = torch.zeros(
        #     (buffer_size, action_dim), dtype=torch.float32, device=device
        # )
        # self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._next_states = torch.zeros(
        #     (buffer_size, state_dim), dtype=torch.float32, device=device
        # )
        # self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        # self._device = device

        self._next_state_embeddings = None
        self._action_embeddings = None
        self._state_embeddings = None
        self.has_embedded = False

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

    def encode_raw_d4rl_data(self, domain: str, dataset: str, tokenizer, language_model, prefix, suffix, batch_size=64,
                             encoding_only=False):
        # Currently do not support language encoding twice.
        if self.has_embedded:
            raise ValueError('Action Denied! ReplayBuffer has contained language embeddings.')

        # Check whether the d4rl raw data has been loaded.
        if self._size == 0:
            raise ValueError('Action Denied! No D4RL data loaded.')

        data = {'states': None, 'next_states': None, 'actions': None}

        # Load saved language embeddings
        folder_path = '/corl/icl4rl/language_embeddings/'
        state_file_name = '%s_%s_%s_%s_%s_S.pkl' % (
            domain, dataset, prefix['state']['nickname'], suffix['state']['nickname'],
            language_model.__class__.__name__)
        action_file_name = '%s_%s_%s_%s_%s_A.pkl' % (
            domain, dataset, prefix['action']['nickname'], suffix['action']['nickname'],
            language_model.__class__.__name__)
        state_file_path = os.path.join(folder_path, state_file_name)
        action_file_path = os.path.join(folder_path, action_file_name)

        print('Attempt to encode data from %s_%s_%s_%s_%s.' % (
            domain, dataset, prefix['state']['nickname'], suffix['state']['nickname'],
            language_model.__class__.__name__))

        if os.path.exists(state_file_path):
            with open(state_file_path, 'rb') as file:
                data['states'], data['next_states'] = pickle.load(file)
            print('State embeddings successfully loaded from: ', state_file_name)
        else:
            print('State embeddings file does not exist. Start encoding instead.')

        if os.path.exists(action_file_path):
            with open(action_file_path, 'rb') as file:
                data['actions'] = pickle.load(file)
            print('Action embeddings successfully loaded from: ', action_file_name)
        else:
            print('Action embeddings file does not exist. Start encoding instead.')

        # TODO: Combine the following two 'if' clauses.
        if data['states'] is None:
            for i in tqdm(range(0, self._size, batch_size), desc='Processing States'):
                idx_ub = min(i + batch_size, self._size)
                batch_s = self._states[i:idx_ub].cpu().numpy()
                batch_s_prime = self._next_states[i:idx_ub].cpu().numpy()
                encoded_s = encode(batch_s, domain, tokenizer, language_model,
                                   prefix['state']['state'], suffix['state']['state'], self._device)
                encoded_s_prime = encode(batch_s_prime, domain, tokenizer, language_model,
                                         prefix['state']['state'], suffix['state']['state'], self._device)
                if data['states'] is None:
                    data['states'] = encoded_s
                else:
                    data['states'] = np.concatenate((data['states'], encoded_s), axis=0)
                if data['next_states'] is None:
                    data['next_states'] = encoded_s_prime
                else:
                    data['next_states'] = np.concatenate((data['next_states'], encoded_s_prime), axis=0)

            with open(state_file_path, 'wb') as file:
                pickle.dump((data['states'], data['next_states']), file)
                print('State encoding successful. Save to: ', state_file_name)

        if data['actions'] is None:
            for i in tqdm(range(0, self._size, batch_size), desc='Processing Actions'):
                idx_ub = min(i + batch_size, self._size)
                batch_a = self._actions[i:idx_ub].cpu().numpy()
                encoded_a = encode(batch_a, domain, tokenizer, language_model, prefix['action']['action'],
                                   suffix['action']['action'], self._device)
                if data['actions'] is None:
                    data['actions'] = encoded_a
                else:
                    data['actions'] = np.concatenate((data['actions'], encoded_a), axis=0)

            with open(action_file_path, 'wb') as file:
                pickle.dump(data['actions'], file)
                print('Action encoding successful. Save to: ', action_file_name)

        if not encoding_only:
            self._state_embeddings = self._to_tensor(data["states"])
            self._action_embeddings = self._to_tensor(data["actions"])
            self._next_state_embeddings = self._to_tensor(data["next_states"])
            self.has_embedded = True
            print('Language embeddings loaded for domain %s' % domain)

    def retain_data_ratio(self, data_ratio):
        # Keep 'data_ratio' many of data in current buffer
        # Check whether the d4rl raw data has been loaded.
        if self._size == 0:
            raise ValueError('Action Denied! No D4RL data loaded.')
        if data_ratio == 1.0:
            return
        np.random.seed(0)
        use_size = int(data_ratio * self._size)
        indices = np.random.choice(self._size, use_size, replace=False)
        self._states = self._states[indices]
        self._next_states = self._next_states[indices]
        self._actions = self._actions[indices]
        self._rewards = self._rewards[indices]
        self._dones = self._dones[indices]
        if self.has_embedded:
            self._state_embeddings = self._state_embeddings[indices]
            self._next_state_embeddings = self._next_state_embeddings[indices]
            self._action_embeddings = self._action_embeddings[indices]
        # TODO: Take '_pointer' more seriously.
        self._size = use_size
        self._pointer = use_size
        print('Retain %f of original buffer data: ', use_size)

    def sample(self, batch_size: int, sample_state_embedding: bool = False,
               sample_action_embedding: bool = False) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        if sample_state_embedding:
            if not self.has_embedded:
                raise ValueError('This buffer has not encoded raw d4rl states.')
            states = self._state_embeddings[indices]
            next_states = self._next_state_embeddings[indices]
        else:
            states = self._states[indices]
            next_states = self._next_states[indices]
        if sample_action_embedding:
            if not self.has_embedded:
                raise ValueError('This buffer has not encoded raw d4rl actions.')
            actions = self._action_embeddings[indices]
        else:
            actions = self._actions[indices]
        rewards = self._rewards[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]
