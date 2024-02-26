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
        'source_domain', '', ['halfcheetah'],
        'target_domain', '', ['walker2d'],

        'source_dataset', '', ['medium'],
        'target_dataset', '', ['medium'],

        'data_ratio', 'R', [0.01, 0.1, 1.0],
        'prefix_name', 'PF', ['mjc_short', 'none'],
        'suffix_name', 'SF', ['mjc_unit', 'none'],

        'enable_language_encoding', 'enc', [True, False],
        'cross_training_mode', 'scr', ['ZeroShot', 'SymCoT', 'None'],

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
    if actual_setting['source_domain'] == actual_setting['target_domain']:
        print(f'Skip setup {setting}. Source and target domains are the same.')
        return
    if actual_setting['source_dataset'] != actual_setting['target_dataset']:
        print(f'Skip setup {setting}. Source and target datasets differ.')
        return

    # exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix='')
    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE

    config.group = '%s2%s-%s2%s' % (env2short[config.source_domain], env2short[config.target_domain],
                                    data2short[config.source_dataset], data2short[config.target_dataset])
    config.name = '_'.join([v+str(actual_setting[k]) for k,v in hyper2logname.items() if v != ''])

    run_TD3_BC(config)


if __name__ == '__main__':
    main()
