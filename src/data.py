from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


DATA_URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
    "c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv"
)


def load_data(source: str | object = DATA_URL) -> tuple[FloatArray, FloatArray]:
    """Charge les données des manchots depuis un chemin local ou une URL."""
    df = pd.read_csv(source).dropna()
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    entrees = df[features].to_numpy()
    sorties = np.column_stack(
        [df["species"].eq(name) for name in class_names]
    ).astype(float)
    return entrees, sorties


def get_sets(
    entrees: FloatArray,
    sorties: FloatArray,
    pourcent_train: int,
    pourcent_valid: int,
    random_state: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Sépare les données en ensembles train/valid/test avec stratification.
    """
    rng = np.random.default_rng(random_state)
    if entrees.ndim != 2 or sorties.ndim != 2:
        raise ValueError("entrees et sorties doivent être des tableaux 2D.")
    if len(entrees) != len(sorties):
        raise ValueError("entrees et sorties doivent avoir le même nombre de lignes.")
    if pourcent_train < 0 or pourcent_valid < 0 or pourcent_train + pourcent_valid > 100:
        raise ValueError("Les pourcentages doivent être compris entre 0 et 100.")

    Xtrain_list, Ytrain_list = [], []
    Xvalid_list, Yvalid_list = [], []
    Xtest_list, Ytest_list = [], []

    n_classes = sorties.shape[1]

    for c in range(n_classes):
        idx_class = np.where(sorties[:, c] == 1)[0]
        rng.shuffle(idx_class)

        n_train = int(len(idx_class) * pourcent_train / 100)
        n_valid = int(len(idx_class) * pourcent_valid / 100)

        idx_train = idx_class[:n_train]
        idx_valid = idx_class[n_train:n_train + n_valid]
        idx_test = idx_class[n_train + n_valid:]

        Xtrain_list.append(entrees[idx_train])
        Ytrain_list.append(sorties[idx_train])
        Xvalid_list.append(entrees[idx_valid])
        Yvalid_list.append(sorties[idx_valid])
        Xtest_list.append(entrees[idx_test])
        Ytest_list.append(sorties[idx_test])

    Xtrain = np.vstack(Xtrain_list)
    Ytrain = np.vstack(Ytrain_list)
    Xvalid = np.vstack(Xvalid_list)
    Yvalid = np.vstack(Yvalid_list)
    Xtest = np.vstack(Xtest_list)
    Ytest = np.vstack(Ytest_list)

    moy = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    std[std == 0] = 1
    Xtrain = (Xtrain - moy) / std
    Xvalid = (Xvalid - moy) / std
    Xtest = (Xtest - moy) / std

    idx_train = rng.permutation(len(Xtrain))
    idx_valid = rng.permutation(len(Xvalid))
    idx_test = rng.permutation(len(Xtest))

    return (
        Xtrain[idx_train],
        Ytrain[idx_train],
        Xvalid[idx_valid],
        Yvalid[idx_valid],
        Xtest[idx_test],
        Ytest[idx_test],
    )
