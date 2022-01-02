import os
import json
import pathlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('ggplot')

matplotlib.rcParams['text.usetex'] =  True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
matplotlib.rcParams.update({'font.size': 13})

NUM_RUNS = 5
NUM_PROCESSORS = 5
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/exp_data/'

EXPS_PATHS = [
    '/home/pedro/git/DECAF/tests/exp_data/MAIN_door_env_2021-12-26-02-07-12/train_data.json',
    '/home/pedro/git/DECAF/tests/exp_data/MAIN_door_env_2021-12-26-13-59-59/train_data.json',
    '/home/pedro/git/DECAF/tests/exp_data/MAIN_door_env_2021-12-26-14-42-15/train_data.json',
]

DEFAULT_EXP_PATH = '/home/pedro/git/DECAF/tests/exp_data/MAIN_door_env_2021-12-27-00-17-07/train_data.json'


FIGURE_X = 6.0
FIGURE_Y = 4.0

def mean_agg_func(samples: np.ndarray, num_resamples: int=25_000):
    """
        Computes mean.
    """
    # Point estimation.
    point_estimate = np.mean(samples)
    # Confidence interval estimation.
    resampled = np.random.choice(samples,
                                size=(len(samples), num_resamples),
                                replace=True)
    point_estimations = np.mean(resampled, axis=0)
    confidence_interval = [point_estimate - np.percentile(point_estimations, 5),
                           np.percentile(point_estimations, 95) - point_estimate]
    return point_estimate, confidence_interval


if __name__ == "__main__":

    exp_data_biased = []
    exp_data_debiased = []

    for exp_path in EXPS_PATHS:
        f = open(exp_path)        
        data = json.load(f)
        data = eval(data) # TODO: Fix this. Use an approriate encoder when writing to json file instead.
        print(data)
        exp_data_biased += [d['exp_data_biased'] for d in data]
        exp_data_debiased += [d['exp_data_debiased'] for d in data]
        f.close()

    print(len(exp_data_biased))
    print(len(exp_data_debiased))

    #df_biased = pd.DataFrame(exp_data_biased)
    #print('df_biased')
    #print(df_biased)
    #print(df_biased.mean())

    df_debiased = pd.DataFrame(exp_data_debiased)
    print('df_debiased')
    print(df_debiased)
    print(df_debiased.mean())
    print(df_debiased.std())

    # Load default data.
    f = open(DEFAULT_EXP_PATH)        
    data = json.load(f)
    data = eval(data) # TODO: Fix this. Use an approriate encoder when writing to json file instead.
    print(data)
    def_data = [d['data_statistics'] for d in data]
    f.close()

    df_default = pd.DataFrame(def_data)
    print('df_default')
    print(df_default)
    print(df_default.mean())
    print(df_default.std())

    fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    X = np.arange(3)

    # Y_true = [covTrue_eps00[env_id][0] for env_id in data.keys()]
    # Y_false = [covFalse_eps00[env_id][0] for env_id in data.keys()]
    # cis_true = [[covTrue_eps00[env_id][1],covTrue_eps00[env_id][2]] for env_id in data.keys()]
    # cis_false = [[covFalse_eps00[env_id][1],covFalse_eps00[env_id][2]] for env_id in data.keys()]
    # cis_true = np.array(cis_true).T
    # cis_false = np.array(cis_false).T

    Ys_a0_mean = []
    Ys_a0_cis = []
    Y_a0_mean, Y_a0_cis = mean_agg_func(df_default['E_do_A0'].to_numpy())
    Ys_a0_mean.append(Y_a0_mean)
    Ys_a0_cis.append(Y_a0_cis)

    Y_a0_mean, Y_a0_cis = mean_agg_func(df_default['E_a0'].to_numpy())
    Ys_a0_mean.append(Y_a0_mean)
    Ys_a0_cis.append(Y_a0_cis)

    Y_a0_mean, Y_a0_cis = mean_agg_func(df_debiased['E_a0'].to_numpy())
    Ys_a0_mean.append(Y_a0_mean)
    Ys_a0_cis.append(Y_a0_cis)

    Ys_a1_mean = []
    Ys_a1_cis = []
    Y_a1_mean, Y_a1_cis = mean_agg_func(df_default['E_do_A1'].to_numpy())
    Ys_a1_mean.append(Y_a1_mean)
    Ys_a1_cis.append(Y_a1_cis)

    Y_a1_mean, Y_a1_cis = mean_agg_func(df_default['E_a1'].to_numpy())
    Ys_a1_mean.append(Y_a1_mean)
    Ys_a1_cis.append(Y_a1_cis)

    Y_a1_mean, Y_a1_cis = mean_agg_func(df_debiased['E_a1'].to_numpy())
    Ys_a1_mean.append(Y_a1_mean)
    Ys_a1_cis.append(Y_a1_cis)

    Ys_a1_cis = np.array(Ys_a1_cis).T
    Ys_a0_cis = np.array(Ys_a0_cis).T
    print(Ys_a0_mean)
    print(Ys_a0_cis)
    print(Ys_a1_mean)
    print(Ys_a1_cis)

    width = 0.3
    rects1 = plt.bar(X - width/2, Ys_a0_mean, width, yerr=Ys_a0_cis, label='$b = b_0$')
    rects2 = plt.bar(X + width/2, Ys_a1_mean, width, yerr=Ys_a1_cis, label='$b = b_1$')

    plt.xticks(X, labels=["$\mathbb{E}[D'|$do$(b);\mathcal{D}_{\pi_b}]$",
                "$\mathbb{E}[D'|b;\mathcal{D}_{\pi_b}]$",
                "$\mathbb{E}[D'|b;\hat{\mathcal{D}}]$ (Our)"])

    plt.ylabel('Expected value')
    #plt.xlabel('Version')
    plt.legend()
    plt.savefig('plot_1.png', bbox_inches='tight', pad_inches=0)
    plt.savefig('plot_1.pdf', bbox_inches='tight', pad_inches=0)

    """ fig = plt.figure()
    fig.set_size_inches(FIGURE_X, FIGURE_Y)

    x_ticks_pos = np.arange(1,len(EXPS_DATA)+1)
    plt.violinplot(errors_list, positions=x_ticks_pos, showextrema=True)

    for i in range(len(EXPS_DATA)):
        plt.scatter([x_ticks_pos[i]]*len(errors_list[i]), errors_list[i],
                        color=GRAY_COLOR, zorder=100, alpha=0.6)

    plt.ylabel(r'$Q$-values error')
    plt.xlabel(r'$\mathbb{E}[\mathcal{H}(\mu)]$')
    plt.xticks(ticks=x_ticks_pos, labels=[e['label_1'] for e in EXPS_DATA])

    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_distribution.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.savefig(f'{PLOTS_FOLDER_PATH}/qvals_final_distribution.png',
                bbox_inches='tight', pad_inches=0)
    plt.close() """