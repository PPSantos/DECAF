from typing import Any, Tuple

import random
import networkx as nx
import numpy as np
import pandas as pd


def gen_door_env_data(N, policy_eps) -> pd.DataFrame:

    d = []
    for _ in range(N):

        # Sample state.
        s_type = np.random.choice([0,1], p=[0.6,0.4])
        if s_type == 0:
            s_t = np.random.uniform()
        else:
            s_t = np.random.uniform() - 1

        # Sample action.
        # a_t = np.random.choice([0,1])

        if s_t <= 0.0:
            a_t = np.random.choice([0,1], p=[0.4,0.6])
        else:
            a_t = np.random.choice([0,1], p=[0.9,0.1])

        # if s_t <= 0.0:
        #     a_t = 1
        # else:
        #     a_t = 0
        # if random.uniform(0,1) < policy_eps:
        #     a_t = np.random.randint(2)

        # Sample reward.
        if s_t <= 0.0 and a_t == 0:
            r_mean = -0.5
        elif s_t <= 0.0 and a_t == 1:
            r_mean = 0.5
        elif s_t > 0.0 and a_t == 0:
            r_mean = 0.5
        elif s_t > 0.0 and a_t == 1:
            r_mean = -0.5
        else:
            raise ValueError('Error generating dataset.')
        r_t = np.random.normal(loc=r_mean, scale=0.1)

        # One-hot encode action.
        one_hot = np.zeros(2) # num actions = 2
        one_hot[a_t] = 1
        a_t = list(one_hot)

        d.append([s_t] + a_t + [r_t])

    return pd.DataFrame(d) #, columns=list(map(str, [0,1,2,3])))


# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
    G: Any,
    base_mean: float = 0,
    base_var: float = 0.3,
    mean: float = 0,
    var: float = 1,
    SIZE: int = 10000,
    err_type: str = "normal",
    perturb: list = [],
    sigmoid: bool = True,
    expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def load_adult() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    df = df.values

    return df.astype(np.uint32)

    #X = df[:, :14].astype(np.uint32)
    #y = df[:, 14].astype(np.uint8)
    #return X, y
