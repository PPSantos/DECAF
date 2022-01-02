import os
import json
import time
import pathlib
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime

import pytorch_lightning as pl
from utils import gen_door_env_data

from learning import GAN
from learning.data import DataModule

NUM_RUNS = 15
NUM_PROCESSORS = 2
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/exp_data/'


def create_exp_name():
    return 'door_env_' + str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def calculate_data_statistics(data):
    d = data.copy()
    one_hot_actions = d.iloc[:, 1:3]
    one_hot_actions.columns = [0,1]
    d['action'] = (one_hot_actions.iloc[:, 0:2] == 1).idxmax(1)
    d['state'] = (d.iloc[:,0] > 0).astype(int)

    # Calculate P(S).
    P_s = d['state'].value_counts() / len(d['state'].index)
    P_s0 = P_s.loc[0]
    P_s1 = P_s.loc[1]

    # Calculate E[R|A].
    E_per_A = d.groupby(['action']).mean()
    E_a0 = E_per_A.loc[0][3]
    E_a1 = E_per_A.loc[1][3]

    # Calculate E[R|do(A)] = E[E_S[R|S,A]].
    E_per_SA = d.groupby(['state', 'action']).mean()
    E_do_A0 = E_per_SA.loc[(0,0)][3]*P_s[0] + E_per_SA.loc[(1,0)][3]*P_s[1]
    E_do_A1 = E_per_SA.loc[(0,1)][3]*P_s[0] + E_per_SA.loc[(1,1)][3]*P_s[1]

    return {'P_s0': P_s0, 'P_s1': P_s1, 'E_a0': E_a0,
            'E_a1': E_a1, 'E_do_A0': E_do_A0, 'E_do_A1': E_do_A1}

def run(args, time_delay):

    time.sleep(time_delay)
    np.random.seed(time_delay)

    # Setup experiment data folder.
    exp_name = create_exp_name()
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)

    # Generate data.
    data = gen_door_env_data(N=args.datasize, policy_eps=args.policy_eps)
    dm = DataModule(data.values)
    data_tensor = dm.dataset.x
    data_statistics = calculate_data_statistics(data)
    print('Data:\n', data)
    print(data_statistics)

    return {'exp_name': exp_name, 'data_statistics': data_statistics}

    # Causal structure.
    variables = {
        0: {'name': 's_t', 'parents': [], 'size': 1, 'type': 'continuous'},
        1: {'name': 'a_t', 'parents': [0], 'size': 2, 'type': 'discrete'},
        2: {'name': 'r_t', 'parents': [0,1], 'size': 1, 'type': 'continuous'},
    }
    topological_order = [0,1,2]

    # Edge removal dictionary.
    bias_dict = {1: [0]}  # This removes edge 0 -> 1.

    # Model initialisation and train.
    model = GAN(
        variables=variables,
        gen_order=topological_order,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_gp=args.lambda_gp,
        d_updates=args.d_updates,
        weight_decay=args.weight_decay,
        l1_g=args.l1_g,
    )
    logger = pl.loggers.TensorBoardLogger(name='tb-logger', save_dir='lightning_logs')
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=20,
        profiler=False,
        #callbacks=[],
    )
    trainer.fit(model, dm)

    # Synthesize data using trained generator (debiasing=On).
    print('-'*20)
    print('Debiasing=On')
    synth_data = (
        model.gen_synthetic(
            data_tensor, gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
        .detach()
        .numpy()
    )
    print('Debiased synthesized data:')
    print(synth_data)

    # Store synthesized data.
    df = pd.DataFrame.from_records(synth_data)
    df.to_csv(exp_path + '/debiased_synthesized_data.csv')

    exp_data_debiased = calculate_data_statistics(df)
    print(exp_data_debiased)

    print('-'*20)
    print('Debiasing=Off')
    # Synthesize data using trained generator (debiasing=off).
    synth_data = (
        model.gen_synthetic(
            data_tensor, gen_order=model.get_gen_order()
        )
        .detach()
        .numpy()
    )
    print('Biased synthesized data:')
    print(synth_data)

    # Store synthesized data.
    df = pd.DataFrame.from_records(synth_data)
    df.to_csv(exp_path + '/biased_synthesized_data.csv')

    exp_data_biased = calculate_data_statistics(df)
    print(exp_data_biased)

    return {'exp_name': exp_name, 'exp_data_biased': exp_data_biased,
            'exp_data_debiased': exp_data_debiased}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--d_updates", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--l1_g", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--datasize", type=int, default=10_000)
    parser.add_argument("--policy_eps", type=float, default=0.0)
    args = parser.parse_args()
    print(args)

    # Setup main experiment data folder.
    exp_name = 'MAIN_' + create_exp_name()
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID (MAIN):', exp_name)

    with mp.Pool(processes=NUM_PROCESSORS) as pool:
        data = pool.starmap(run, [(args, 1*t) for t in range(NUM_RUNS)])
        pool.close()
        pool.join()
    print(data)

    # Store train log data.
    f = open(exp_path + "/train_data.json", "w")
    dumped = json.dumps(data) #, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()

    print('\nExperiment ID (MAIN):', exp_name)
    print('\nData biased:')
    data_biased = pd.DataFrame([run['exp_data_biased'] for run in data])
    print(data_biased)
    print(data_biased.mean())

    print('\nData debiased:')
    data_debiased = pd.DataFrame([run['exp_data_debiased'] for run in data])
    print(data_debiased)
    print(data_debiased.mean())
