import os
import pickle
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

    def to_tensor(self, batch) -> List[torch.Tensor]:
        return [torch.tensor(batch[k], dtype=torch.float32, device=self.config.device)
                for k in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']]

    def get_sas(self, idx, split='target', use_state_emb=False, use_action_emb=False):
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
            'next_observations': raw_dataset['next_observations'][idx]
        }
        if use_state_emb:
            batch['observations'] = emb_dataset['observations'][idx]
            batch['next_observations'] = emb_dataset['next_observations'][idx]
        if use_action_emb:
            batch['actions'] = emb_dataset['actions'][idx]

        return batch

    def sample(self):
        batch_size = self.config.batch_size
        add_concat = self.config.add_concat
        enable_emb = self.config.enable_emb
        mode = self.config.cross_train_mode

        if mode == 'None':
            idx = np.random.choice(self.target_size, size=batch_size, replace=False)
            batch = self.get_sas(idx, split='target', use_state_emb=enable_emb)
            if enable_emb and add_concat:
                batch_raw = self.get_sas(idx, split='target', use_state_emb=False)
                batch = {k: np.concatenate((v, batch_raw[k]), axis=1) for k, v in batch.items()}

            batch['rewards'] = self.target_raw['rewards'][idx][..., None]
            batch['dones'] = self.target_raw['terminals'][idx][..., None]
            return self.to_tensor(batch)
        else:
            raise NotImplementedError('Have not implemented such cross training mode in Buffer.sample method.')
