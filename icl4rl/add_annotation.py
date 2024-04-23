import os
import pickle
import joblib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from icl4rl.state_action_annotations import ALL_ANNOTATIONS_DICT

TensorBatch = List[torch.Tensor]


def add_annotation(raw_vectors: np.ndarray, prefix_annotations: list, suffix_annotations: list):
    # Handle one-dim state during inference
    if raw_vectors.ndim == 1:
        raw_vectors = raw_vectors.reshape(1, -1)
    if not (len(prefix_annotations) == len(suffix_annotations) == raw_vectors.shape[1]):
        raise ValueError("The annotation lists and the dimension of raw vectors must have the same length.")
    prefix_list = [s + ': ' if s != '' else '' for s in prefix_annotations]
    suffix_list = [' ' + s + ',' if s != '' else ',' for s in suffix_annotations]
    raw_vectors = raw_vectors.round(decimals=5).astype('str')
    readable = np.char.add(np.char.add(prefix_list, raw_vectors), suffix_list)
    readable = np.apply_along_axis(' '.join, 1, readable).tolist()
    return readable


def LM_mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@torch.no_grad()
def encode_fn(raw_vectors: np.ndarray, domain_name: str, tokenizer, language_model,
              prefix_annotation_dict: Optional[Dict[str, list]],
              suffix_annotation_dict: Optional[Dict[str, list]], emb_mode: str = 'avg', device: str = 'cpu'):
    if prefix_annotation_dict.get(domain_name) is None:
        raise KeyError('Unsupported domain prefix annotation.')
    if suffix_annotation_dict.get(domain_name) is None:
        raise KeyError('Unsupported domain suffix annotation.')

    prefix_annotations = prefix_annotation_dict[domain_name]
    suffix_annotations = suffix_annotation_dict[domain_name]
    readable = add_annotation(raw_vectors, prefix_annotations, suffix_annotations)
    lm_inputs = tokenizer(readable, padding=True, truncation=True, return_tensors='pt').to(device)
    lm_outputs = language_model(**lm_inputs)
    if emb_mode == 'avg':
        embeddings = LM_mean_pooling(lm_outputs, lm_inputs['attention_mask'])
    elif emb_mode == 'cls':
        embeddings = lm_outputs[0][:, 0, :]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


class Buffer:
    def __init__(self, config, source_domain=None, target_domain=None):
        if (source_domain or target_domain) is None:
            raise ValueError('Either domain needs to be not None type.')

        self.source_raw, self.source_emb, self.source_size = None, None, 0
        self.target_raw, self.target_emb, self.target_size = None, None, 0

        if source_domain:
            self.source_raw = source_domain.raw_dataset
            self.source_emb = source_domain.emb_dataset
            self.source_size = self.source_raw['observations'].shape[0]
        if target_domain:
            self.target_raw = target_domain.raw_dataset
            self.target_emb = target_domain.emb_dataset
            self.target_size = self.target_raw['observations'].shape[0]

        self.config = config

        if config.cross_train_mode == 'MDPFD':
            self.mdp_data = self.load_mdp_data()
            self.mdp_size = self.mdp_data['observations'].shape[0]
            state_dim = target_domain.state_dim
            if config.enable_emb:
                state_dim = self.target_emb['observations'].shape[1]
                if config.add_concat:
                    state_dim += target_domain.state_dim
            np.random.seed(0)
            index2state = 2 * np.random.rand(100, state_dim) - 1
            index2action = 2 * np.random.rand(100, target_domain.action_dim) - 1
            self.idx2state, self.idx2action = index2state.astype(np.float32), index2action.astype(np.float32)

    def load_mdp_data(self):
        folder_path = self.config.emb_save_folder
        file_name = 'mdp_traj1000_ns100_na100_pt1_tt1.pkl'
        file_path = os.path.join(folder_path, 'mdp_data', file_name)

        print("MDP pretrain data loaded from:", file_name)
        mdp_dataset = joblib.load(file_path)

        return mdp_dataset

    def to_tensor(self, batch) -> Dict:
        for k, v in batch.items():
            batch[k] = torch.tensor(v, dtype=torch.float32, device=self.config.device)

        return batch

    def get_batch(self, idx, split='target', use_state_emb=False, use_action_emb=False, add_concat=False):
        if split == 'mdp':
            if self.config.cross_train_mode != 'MDPFD':
                raise ValueError('The current training mode is not MDP pretraining. No MDP data loaded.')

            batch = {
                'observations': self.mdp_data['observations'][idx],
                'actions': self.mdp_data['actions'][idx],
                'next_observations': self.mdp_data['next_observations'][idx],
            }
            batch['observations'] = self.idx2state[batch['observations']]
            batch['next_observations'] = self.idx2state[batch['next_observations']]
            batch['actions'] = self.idx2action[batch['actions']]

            return batch

        if split == 'target':
            raw_dataset = self.target_raw
            emb_dataset = self.target_emb
        elif split == 'source':
            raw_dataset = self.source_raw
            emb_dataset = self.source_emb
        else:
            raise KeyError('The \'split\' argument should be either \'target\' or \'source\'.')

        if (raw_dataset or emb_dataset) is None:
            raise ValueError('The corresponding dataset is not in the buffer.')

        batch = {
            'observations': raw_dataset['observations'][idx],
            'actions': raw_dataset['actions'][idx],
            'next_observations': raw_dataset['next_observations'][idx],
            'rewards': raw_dataset['rewards'][idx][..., None],
            'terminals': raw_dataset['terminals'][idx][..., None],
        }
        if use_state_emb:
            for k in ['observations', 'next_observations']:
                if add_concat:
                    batch[k] = np.concatenate((emb_dataset[k][idx], batch[k]), axis=1)
                else:
                    batch[k] = emb_dataset[k][idx]

        if use_action_emb:
            batch['actions'] = emb_dataset['actions'][idx]

        return batch

    def sample(self, pretrain: bool = False):
        batch_size = self.config.batch_size
        add_concat = self.config.add_concat
        enable_emb = self.config.enable_emb
        mode = self.config.cross_train_mode

        if pretrain:
            if mode == 'SelfFD':
                mode = 'None'
            elif mode == 'CrossFD':
                mode = 'ZeroShot'
        else:
            mode = 'None' if mode in ['SelfFD', 'CrossFD', 'MDPFD'] else mode

        if mode == 'None':
            idx = np.random.choice(self.target_size, size=batch_size, replace=False)
            batch = self.get_batch(idx, split='target', use_state_emb=enable_emb, add_concat=add_concat)

        elif mode == 'ZeroShot':
            idx = np.random.choice(self.source_size, size=batch_size, replace=False)
            batch = self.get_batch(idx, split='source', use_state_emb=enable_emb, add_concat=add_concat)

        elif mode == 'MDPFD':
            idx = np.random.choice(self.mdp_size, size=batch_size, replace=False)
            batch = self.get_batch(idx, split='mdp')

        elif mode == 'RandomCoT':
            total_size = self.source_size + self.target_size
            idx = np.random.choice(total_size, size=batch_size, replace=False)
            source_idx = idx[idx < self.source_size]
            target_idx = idx[idx >= self.source_size] - self.source_size
            source_batch = self.get_batch(source_idx, split='source', use_state_emb=enable_emb, add_concat=add_concat)
            target_batch = self.get_batch(target_idx, split='target', use_state_emb=enable_emb, add_concat=add_concat)
            batch = {k: np.concatenate((source_batch[k], target_batch[k]), axis=0) for k in source_batch.keys()}

        else:
            raise NotImplementedError('Have not implemented such cross training mode in Buffer.sample method.')

        return self.to_tensor(batch)
