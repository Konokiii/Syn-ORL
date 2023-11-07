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
from offline.iql import TrainConfig, run_IQL
import argparse
from offline.utils import *

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

    exp_prefix = 'iql'
    settings = [
        'env', '', MUJOCO_3_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'pretrain_mode', 'preM', ['none'],
        'seed', '', list(range(3, 10))
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)

    # config = pyrallis.load(TrainConfig, '/configs/offline/iql/%s/%s_v2.yaml'
    #                        % (actual_setting['env'], actual_setting['dataset'].replace('-', '_')))

    """replace values"""
    config = TrainConfig(**actual_setting)
    config.device = DEVICE
    if config.env == 'hopper':
        if 'iql_deterministic' not in settings:
            if config.dataset in ['medium-replay', 'medium']:
                config.iql_deterministic = True
        if 'beta' not in settings:
            if config.dataset == 'medium-expert':
                config.beta = 6.0
        if 'iql_tau' not in settings:
            if config.dataset == 'medium-expert':
                config.iql_tau = 0.5

    data_dir = '/train_logs'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, config.seed, data_dir)
    outdir = logger_kwargs["output_dir"]
    exp_name = logger_kwargs["exp_name"]
    run_IQL(config, outdir, exp_name)


if __name__ == '__main__':
    main()
