import os

# with new logger
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress the warning about tokenizer parallelism when eval_actor??

import torch
from dataclasses import replace
import pyrallis
from algorithms.offline.td3_bc import TrainConfig, run_TD3_BC
import argparse
from icl4rl.state_action_annotations import *
from algorithms.offline.utils import *

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    setting = args.setting

    settings = [
        'encoding_only', '', [True],
        'enc_batch_size', '', [128],
        'source_domain', '', ['halfcheetah', 'walker2d', 'hopper'],
        'source_dataset', '', ['medium-expert', 'medium', 'medium-replay'],
        'prefix_annotation', '', [MUJOCO_SHORT_DESCRIPTION, NONE],
        'suffix_annotation', '', [MUJOCO_UNIT, NONE]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')

    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE

    run_TD3_BC(config)


if __name__ == '__main__':
    main()