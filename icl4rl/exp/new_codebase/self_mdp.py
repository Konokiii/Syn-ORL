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
        # 'source_domain', '', ['halfcheetah'],
        'target_domain', '', ['hopper', 'walker2d', 'halfcheetah'],

        # 'source_dataset', '', ['medium'],
        'target_dataset', '', ['medium-replay', 'medium'],

        'enable_emb', 'enc', [True, False],
        'prefix', 'PF', ['mjc_re'],
        'suffix', 'SF', ['mjc_unit'],
        'normalize_emb', 'normE', [False],
        'add_concat', 'cat', [True, False],

        'cross_train_mode', 'M', ['SelfFD', 'MDPFD'],
        'max_pretrain_steps', 'T', [int(1e5)],
        'data_ratio', 'R', [0.1, 0.5, 1.0],

        'hidden_arch', 'arch', ['256-256'],
        'seed', 'S', [0, 1, 2]
    ]

    env2short = {'halfcheetah': 'hal',
                 'walker2d': 'w',
                 'hopper': 'hop'}
    data2short = {'medium': 'm',
                  'medium-replay': 'mr',
                  'medium-expert': 'me'}

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    # Terminate undesirable setups
    if not actual_setting['enable_emb'] and actual_setting['add_concat']:
        print(f'Skip setup {setting}. Repeated exps for no embedding + add concat.')
        return
    # if actual_setting['source_domain'] == actual_setting['target_domain']:
    #     print(f'Skip setup {setting}. Source and target domains are the same.')
    #     return
    # if actual_setting['source_dataset'] != actual_setting['target_dataset']:
    #     print(f'Skip setup {setting}. Source and target datasets differ.')
    #     return
    # if not actual_setting['enable_language_encoding']:
    #     if actual_setting['prefix_name'] == actual_setting['suffix_name'] == 'none':
    #         pass
    #     else:
    #         print(f'Skip setup {setting}. Disable language encoding.')
    #         return
    # if actual_setting['cross_training_mode'] == 'ZeroShot':
    #     if actual_setting['data_ratio'] == 1.0:
    #         pass
    #     else:
    #         print(f'Skip setup {setting}. ZeroShot training does not involve source data.')
    #         return

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    # TODO: Run faster exps.
    config.eval_freq = int(1e4)
    config.n_episodes = 5

    config.group = 'new_code'
    config.name = '_'.join([v+str(actual_setting[k]) for k,v in hyper2logname.items() if v != ''])

    run_TD3_BC(config)


if __name__ == '__main__':
    main()
