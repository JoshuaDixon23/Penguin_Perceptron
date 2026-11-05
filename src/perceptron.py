from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence


FloatArray = NDArray[np.float64]


def f_activation(x: float, seuil: float) -> int:
    """
    Fonction d'activation binaire (me permet de changer la focntion d'activation facilement si besoin)
    Parameters:
    -----------
    x : valeur d'entrée
    seuil : seuil d'activation
    """
    return 1 if x > seuil else 0


def neur(
    li_e: Sequence[float],
    li_poid: Sequence[float],
    seuil: float,
) -> int:
    """
    Calcule la sortie d'un neurone après application de la fonction d'activation.
    Parameters:
    -----------
    li_e : liste des entrées
    li_poid : liste des poids
    seuil : seuil d'activation
    """
    if len(li_e) != len(li_poid):
        raise ValueError("Les entrées et les poids doivent avoir la même longueur.")
    res = 0
    for i in range(len(li_e)):
        res += li_e[i] * li_poid[i]
    return f_activation(res, seuil)


def apprentissage(
    Xapp: FloatArray,
    Yapp: FloatArray,
    lr: float,
    etapes: int,
    seuil: float,
    seed: int | None = None,
) -> FloatArray:
    """
    Apprentissage d'un perceptron multi-classes avec la règle de Hebb modifiée.
    Parameters:
    -----------
    Xapp : matrice des entrées d'apprentissage
    Yapp : matrice des sorties d'apprentissage
    lr : taux d'apprentissage
    etapes : nombre d'étapes d'apprentissage
    seuil : seuil d'activation
    seed : graine utilisée pour l'initialisation des poids

    Returns:
    --------
    li_poid : liste des poids appris
    """
    nb_neur = 3
    if Xapp.ndim != 2 or Xapp.shape[1] != 4:
        raise ValueError("Xapp doit avoir la forme (n_samples, 4).")
    if Yapp.ndim != 2 or Yapp.shape != (Xapp.shape[0], nb_neur):
        raise ValueError("Yapp doit avoir la forme (n_samples, 3).")

    rng = np.random.default_rng(seed)
    li_poid = rng.random((3, 4))

    for k in range(etapes):
        total_e = 0
        for i in range(len(Xapp)):
            li_e = [Xapp[i][0], Xapp[i][1], Xapp[i][2], Xapp[i][3]]
            for j in range(nb_neur): # nb neuronnes
                nr_poid = li_poid[j]
                #calcul de la sortie
                s = neur(li_e, nr_poid, seuil)
                y = Yapp[i][j]
                e = y - s

                total_e += abs(e)
                if y != s:
                    for l in range(len(nr_poid)):
                        nr_poid[l] = nr_poid[l] + (lr * e * li_e[l])

    return li_poid


def neur_score(li_e: Sequence[float], li_poid: Sequence[float]) -> float:
    """
    Calcule le score brut (avant activation) du neurone.

    Parameters: 
    -----------
    li_e : liste des entrées
    li_poid : liste des poids
    """
    if len(li_e) != len(li_poid):
        raise ValueError("Les entrées et les poids doivent avoir la même longueur.")
    res = 0
    for i in range(len(li_e)):
        res += li_e[i] * li_poid[i]
    return res
