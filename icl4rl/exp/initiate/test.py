import os

# with new logger
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'

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

    exp_prefix = 'test'
    settings = [
        'source_dataset', '', ['medium-expert'],
        'target_dataset', '', ['medium-expert'],
        'data_ratio', 'ratio', [0.5],
        'enable_source_domain', 'src', [True, False],
        'enable_language_encoding', 'enc', [True, False],
        'cross_training_mode', 'mode', ['ZeroShot'],
        'seed', '', [0, 1]
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)

    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    config.eval_freq = int(2)
    config.n_episodes = 1
    config.max_timesteps = 4
    config.group = 'test_group'
    config.name = exp_name_full

    run_TD3_BC(config)


if __name__ == '__main__':
    main()
