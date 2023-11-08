from quick_plot_helper import quick_plot_with_full_name
from log_alias import *
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

# Global variables:
MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah', ]
MUJOCO_4_ENVS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
MUJOCO_3_DATASETS = ['medium', 'medium-replay', 'medium-expert']
d4rl_9_datasets_envs = ['%s_%s' % (e, d) for e in MUJOCO_3_ENVS for d in MUJOCO_3_DATASETS]
d4rl_12_datasets_envs = ['%s_%s' % (e, d) for e in MUJOCO_4_ENVS for d in MUJOCO_3_DATASETS]

d4rl_test_performance_col_name = 'TestEpNormRet'
d4rl_x_axis_col_name = 'Steps'

default_performance_smooth = 5
font_size = 10

data_path = '../train_logs/iql_baseline/'
save_path = './figures/'


def get_full_names_with_envs(base_names, envs):
    # envs can be offline or online envs, offline envs will need to also have dataset name
    n = len(base_names)
    to_return = []
    for i in range(n):
        new_list = []
        for env in envs:
            full_name = base_names[i] + '_' + env
            new_list.append(full_name)
        to_return.append(new_list)
    return to_return


def plot_iql_performance_curves(labels, base_names, env_names):
    y = d4rl_test_performance_col_name

    # aggregate
    quick_plot_with_full_name(
        labels,
        get_full_names_with_envs(base_names, env_names),
        save_name_prefix='agg-iql',
        base_data_folder_path=data_path,
        save_folder_path=save_path,
        y_value=[y],
        x_to_use=d4rl_x_axis_col_name,
        ymax=None,
        smooth=default_performance_smooth,
        axis_font_size=font_size
    )

    # separate
    for i, env_dataset_name in enumerate(env_names):
        quick_plot_with_full_name(  # labels, folder name prefix, envs
            labels,
            get_full_names_with_envs(base_names, [env_dataset_name]),
            save_name_prefix='ind-iql',
            base_data_folder_path=data_path,
            save_folder_path=save_path,
            y_value=[y],
            x_to_use=d4rl_x_axis_col_name,
            ymax=None,
            save_name_suffix=env_dataset_name,
            smooth=default_performance_smooth,
            axis_font_size=font_size
        )

    # separate into one
    if len(env_names) == 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(18, 12))
    for i, env in enumerate(env_names):
        ax = axs[i // ncols][i % ncols]
        ax.set_axis_off()
        path = save_path + f'TestEpNormRet/ind-iql_{y}_{env}.png'
        ax.imshow(pltimg.imread(path))
        ax.set_title(env)
    fig.tight_layout()
    plt.savefig(save_path+f'TestEpNormRet/{nrows}x{ncols}-iql_TestEpNormRet.png')


labels = [
    'IQL-1',
    'IQL-1_Q',
    'IQL-1_V',
    'IQL-1_QV',
]

base_names = [
    iql1_baseline,
    iql1_mdp_q,
    iql1_mdp_v,
    iql1_mdp_qv
]

env_names = d4rl_12_datasets_envs
plot_iql_performance_curves(labels, base_names, env_names)
