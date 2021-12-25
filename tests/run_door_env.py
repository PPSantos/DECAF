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

NUM_RUNS = 3
NUM_PROCESSORS = 3
DATA_FOLDER_PATH = str(pathlib.Path(__file__).parent.absolute()) + '/exp_data/'


def create_exp_name():
    return 'door_env_' + str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))

def main(time_delay):

    time.sleep(time_delay)
    np.random.seed(time_delay)

    # Setup experiment data folder.
    exp_name = create_exp_name()
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID:', exp_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=8)
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

    # Generate data.
    data = gen_door_env_data(N=args.datasize, policy_eps=args.policy_eps)
    dm = DataModule(data.values)
    data_tensor = dm.dataset.x
    print('Data:')
    print(data)

    # Calculate statistical quantities on original data.
    orig_data = data.copy()
    one_hot_actions = orig_data.iloc[:, 1:3]
    one_hot_actions.columns = [0,1]
    orig_data['action'] = (one_hot_actions.iloc[:, 0:2] == 1).idxmax(1)
    orig_data['state'] = (orig_data.iloc[:,0] > 0).astype(int)
    print(orig_data)

    # Calculate P(S).
    print('P(s):')
    P_s = orig_data['state'].value_counts() / len(orig_data['state'].index)
    print(P_s)

    # Calculate E[R|A].
    print('E[R|A]:')
    E_per_A = orig_data.groupby(['action']).mean()
    print(E_per_A)

    # Calculate E[R|do(A)] = E[E_S[R|S,A]].
    print('E[R|do(A)] = E[E_S[R|S,A]]:')
    E_per_SA = orig_data.groupby(['state', 'action']).mean()
    print(E_per_SA)
    E_do_A0 = E_per_SA.loc[(0,0)][3]*P_s[0] + E_per_SA.loc[(1,0)][3]*P_s[1]
    E_do_A1 = E_per_SA.loc[(0,1)][3]*P_s[0] + E_per_SA.loc[(1,1)][3]*P_s[1]
    print(E_do_A0)
    print(E_do_A1)

    # Causal structure.
    variables = {
        0: {'name': 's_t', 'parents': [], 'size': 1, 'type': 'continuous'},
        1: {'name': 'a_t', 'parents': [0], 'size': 2, 'type': 'discrete'},
        2: {'name': 'r_t', 'parents': [0,1], 'size': 1, 'type': 'continuous'},
    }
    topological_order = [0,1,2]

    # Edge removal dictionary.
    #bias_dict = {}
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

    synth_data = df.copy()
    one_hot_actions = synth_data.iloc[:, 1:3]
    one_hot_actions.columns = [0,1]
    synth_data['action'] = (one_hot_actions.iloc[:, 0:2] == 1).idxmax(1)
    synth_data['state'] = (synth_data.iloc[:,0] > 0).astype(int)

    # Calculate P(S).
    P_s = synth_data['state'].value_counts() / len(synth_data['state'].index)
    print(P_s)
    P_s0 = P_s.loc[0]
    P_s1 = P_s.loc[1]

    # Calculate E[R|A].
    print('E[R|A]:')
    E_per_A = synth_data.groupby(['action']).mean()
    print(E_per_A)
    E_a0 = E_per_A.loc[0][3]
    E_a1 = E_per_A.loc[1][3]

    # Calculate E[R|do(A)] = E[E_S[R|S,A]].
    print('E[R|do(A)] = E[E_S[R|S,A]]:')
    E_per_SA = synth_data.groupby(['state', 'action']).mean()
    print(E_per_SA)
    E_do_A0 = E_per_SA.loc[(0,0)][3]*P_s[0] + E_per_SA.loc[(1,0)][3]*P_s[1]
    E_do_A1 = E_per_SA.loc[(0,1)][3]*P_s[0] + E_per_SA.loc[(1,1)][3]*P_s[1]
    print(E_do_A0)
    print(E_do_A1)

    exp_data_debiased = {'P_s0': P_s0, 'P_s1': P_s1, 'E_a0': E_a0,
                'E_a1': E_a1, 'E_do_A0': E_do_A0, 'E_do_A1': E_do_A1}

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

    synth_data = df.copy()
    one_hot_actions = synth_data.iloc[:, 1:3]
    one_hot_actions.columns = [0,1]
    synth_data['action'] = (one_hot_actions.iloc[:, 0:2] == 1).idxmax(1)
    synth_data['state'] = (synth_data.iloc[:,0] > 0).astype(int)
    print(synth_data)

    # Calculate P(S).
    print('P(s):')
    P_s = synth_data['state'].value_counts() / len(synth_data['state'].index)
    print(P_s)
    P_s0 = P_s.loc[0]
    P_s1 = P_s.loc[1]

    # Calculate E[R|A].
    print('E[R|A]:')
    E_per_A = synth_data.groupby(['action']).mean()
    print(E_per_A)
    E_a0 = E_per_A.loc[0][3]
    E_a1 = E_per_A.loc[1][3]

    # Calculate E[R|do(A)] = E[E_S[R|S,A]].
    print('E[R|do(A)] = E[E_S[R|S,A]]:')
    E_per_SA = synth_data.groupby(['state', 'action']).mean()
    print(E_per_SA)
    E_do_A0 = E_per_SA.loc[(0,0)][3]*P_s[0] + E_per_SA.loc[(1,0)][3]*P_s[1]
    E_do_A1 = E_per_SA.loc[(0,1)][3]*P_s[0] + E_per_SA.loc[(1,1)][3]*P_s[1]
    print(E_do_A0)
    print(E_do_A1)

    exp_data_biased = {'P_s0': P_s0, 'P_s1': P_s1, 'E_a0': E_a0,
                'E_a1': E_a1, 'E_do_A0': E_do_A0, 'E_do_A1': E_do_A1}

    return {'exp_name': exp_name, 'exp_data_biased': exp_data_biased,
            'exp_data_debiased': exp_data_debiased}

if __name__ == "__main__":

    # Setup experiment data folder.
    exp_name = 'MAIN_' + create_exp_name()
    exp_path = DATA_FOLDER_PATH + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print('\nExperiment ID (MAIN):', exp_name)

    with mp.Pool(processes=NUM_PROCESSORS) as pool:
        data = pool.map(main, [(2*t) for t in range(NUM_RUNS)])
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
