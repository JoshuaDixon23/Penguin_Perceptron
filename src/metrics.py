from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .perceptron import neur, neur_score

FloatArray = NDArray[np.float64]


def matrice_confusion(Ytrue: FloatArray, Ypred: FloatArray) -> NDArray[np.int_]:
    """
    Calcule la matrice de confusion à partir des vraies étiquettes et des étiquettes prédites.
    Parameters:
    -----------
    Ytrue : vraies étiquettes
    Ypred : étiquettes prédites
    Returns:
    --------
    conf_matrix : matrice de confusion
    """
    if Ytrue.ndim != 2 or Ypred.ndim != 2 or Ytrue.shape != Ypred.shape:
        raise ValueError("Ytrue et Ypred doivent avoir la même forme 2D.")
    n_classes = Ytrue.shape[1]
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(Ytrue)):
        true_class = np.argmax(Ytrue[i])
        pred_class = np.argmax(Ypred[i])
        conf_matrix[true_class, pred_class] += 1
    return conf_matrix


def taux_bonne_classification(matrice_conf: NDArray[np.int_]) -> float:
    """
    Calcule le taux de bonne classification à partir de la matrice de confusion.

    Parameters:
    -----------
    matrice_conf : matrice de confusion

    Returns:
    --------
    taux : taux de bonne classification
    """
    nb_ech = 0
    trace = 0
    for i, line in enumerate(matrice_conf):
        for row in range(len(line)):
            nb_ech += matrice_conf[i][row]
            if i == row:
                trace += matrice_conf[i][row]
    print("taux de bonne classification :", trace/nb_ech)
    return trace/nb_ech


def test(
    Xtest: FloatArray,
    Ytest: FloatArray,
    li_poid: FloatArray,
    seuil: float,
) -> NDArray[np.float64]:
    """
    Teste le modèle sur l'ensemble de test et retourne la matrice de confusion.

    Parameters:
    -----------
    Xtest : liste des entrées de test
    Ytest : liste des sorties de test
    li_poid : liste des poids du modèle entraîné
    seuil : seuil d'activation du neurone
    """
    Ypred = np.zeros_like(Ytest)
    for i in range(len(Xtest)):
        li_e = [Xtest[i][0], Xtest[i][1], Xtest[i][2], Xtest[i][3]]
        for j in range(len(li_poid)):
            nr_poid = li_poid[j]
            s = neur(li_e, nr_poid, seuil)
            Ypred[i][j] = s

    conf_matrix = matrice_confusion(Ytest, Ypred)

    return conf_matrix


def compute_mauc(
    Xtest: FloatArray,
    Ytest: FloatArray,
    li_poid: FloatArray,
    seuil: float = 0,
) -> float:
    """Calcule le MAUC moyen pour les trois classes."""
    del seuil
    aucs = []
    for espece in range(Ytest.shape[1]):
        specificites, sensibilites, _ = roc_espece(
            li_poid, Xtest, Ytest, espece=espece, step=0.01
        )
        aucs.append(calculer_auc(specificites, sensibilites))
    return float(np.mean(aucs))


def roc_espece(
    li_poid_trained: FloatArray,
    Xtest: FloatArray,
    Ytest: FloatArray,
    espece: int = 0,
    step: float = 0.1,
) -> tuple[list[float], list[float], NDArray[np.float64]]:
    """
    Calcule la sensibilité et la spécificité pour différents seuils
    pour UNE SEULE espèce.
    
    Parameters:
    -----------
    li_poid_trained : array des poids entraînés
    Xtest : données de test
    Ytest : labels de test
    espece : int, 0=Adelie, 1=Gentoo, 2=Chinstrap
    step : pas pour les seuils
    
    Returns:
    --------
    specificites : list des spécificités pour chaque seuil
    sensibilites : list des sensibilités pour chaque seuil
    seuils : list des seuils testés
    """
    sensibilites = []
    specificites = []
    
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    
    # Calculer les scores bruts pour cette espèce
    scores_test = []
    for i in range(len(Xtest)):
        li_e = [Xtest[i][0], Xtest[i][1], Xtest[i][2], Xtest[i][3]]
        score = neur_score(li_e, li_poid_trained[espece])
        scores_test.append(score)
    
    # le ranger dans un array
    scores_test = np.array(scores_test)
    
    # Trouver la plage des scores (seuil min et max)
    min_score = np.min(scores_test)
    max_score = np.max(scores_test)
    seuils = np.arange(min_score - 0.5, max_score + 0.5, step)
    
    for seuil_test in seuils:
        # Prédictions binaires pour cette espèce avec ce seuil
        # Verifie si le resultat du test est superieur au seuil et le convertit en 0 ou 1
        Ypred_espece = (scores_test > seuil_test).astype(int)
        # Ytest contient 3 colonnes (une par espèce), on prend la colonne 'espece' et le convertit en 0 ou 1
        Ytrue_espece = Ytest[:, espece].astype(int)
        
        # Calculer VP, VN, FP, FN
        VP = np.sum((Ytrue_espece == 1) & (Ypred_espece == 1))
        VN = np.sum((Ytrue_espece == 0) & (Ypred_espece == 0))
        FP = np.sum((Ytrue_espece == 0) & (Ypred_espece == 1))
        FN = np.sum((Ytrue_espece == 1) & (Ypred_espece == 0))
        
        # Sensibilité = VP / (VP + FN)
        if (VP + FN) > 0:
            sensibilite = VP / (VP + FN)
        else:
            sensibilite = 0
        
        # Spécificité = VN / (VN + FP)
        if (VN + FP) > 0:
            specificite = VN / (VN + FP)
        else:
            specificite = 0
        
        sensibilites.append(sensibilite)
        specificites.append(specificite)
    
    return specificites, sensibilites, seuils


def find_seuil_optimal(
    specificites: list[float],
    sensibilites: list[float],
    seuils: NDArray[np.float64],
) -> tuple[np.float64, np.float64, np.float64]:
    """
    Trouve le seuil où sensibilité et spécificité sont les plus proches.

    Parameters:
    -----------
    specificites : list des spécificités
    sensibilites : list des sensibilités
    seuils : list des seuils
    """
    specificites = np.array(specificites)
    sensibilites = np.array(sensibilites)
    seuils = np.array(seuils)
    
    distances = np.abs(sensibilites - specificites)
    idx_optimal = np.argmin(distances)
    
    seuil_optimal = seuils[idx_optimal]
    return seuil_optimal, 1 - specificites[idx_optimal], sensibilites[idx_optimal]


def point_roc_pour_seuil(
    li_poid: FloatArray,
    Xtest: FloatArray,
    Ytest: FloatArray,
    espece: int,
    seuil: float,
) -> tuple[np.float64, np.float64]:
    """Retourne (1 - spécificité, sensibilité) pour un seuil imposé."""
    scores = np.array([neur_score(x, li_poid[espece]) for x in Xtest])
    ytrue = Ytest[:, espece].astype(int)
    ypred = (scores > seuil).astype(int)

    vp = np.sum((ytrue == 1) & (ypred == 1))
    vn = np.sum((ytrue == 0) & (ypred == 0))
    fp = np.sum((ytrue == 0) & (ypred == 1))
    fn = np.sum((ytrue == 1) & (ypred == 0))
    sensibilite = vp / (vp + fn) if vp + fn else 0
    specificite = vn / (vn + fp) if vn + fp else 0
    return 1 - specificite, sensibilite


def calculer_auc(
    specificites: list[float] | NDArray[np.float64],
    sensibilites: list[float] | NDArray[np.float64],
) -> float:
    """
    Calcule l'AUC (Area Under the Curve) pour une courbe ROC.
    
    Parameters:
    -----------
    specificites : list ou array des spécificités
    sensibilites : list ou array des sensibilités
    
    Returns:
    --------
    auc : float, aire sous la courbe ROC (entre 0 et 1)
    """
    # Convertir en arrays numpy
    specificites = np.array(specificites)
    sensibilites = np.array(sensibilites)
    
    # Calculer 1 - spécificité pour l'axe X de la courbe ROC
    fpr = 1 - specificites  # False Positive Rate
    tpr = sensibilites      # True Positive Rate
    
    # Trier par FPR croissant pour intégration correcte
    indices_tri = np.argsort(fpr)
    fpr_trie = fpr[indices_tri]
    tpr_trie = tpr[indices_tri]
    
    # Calculer l'aire sous la courbe avec la règle des trapèzes
    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc = abs(trapezoid(tpr_trie, fpr_trie))
    
    return auc


def calculer_ap(
    precisions: list[float] | NDArray[np.float64],
    rappels: list[float] | NDArray[np.float64],
) -> float:
    """
    Calcule l'AP (Average Precision) pour une courbe Précision-Rappel.
    
    Parameters:
    -----------
    precisions : list ou array des précisions
    rappels : list ou array des rappels
    
    Returns:
    --------
    ap : float, aire sous la courbe Précision-Rappel (entre 0 et 1)
    """
    # Convertir en arrays numpy
    precisions = np.array(precisions)
    rappels = np.array(rappels)
    
    # Trier par rappel croissant pour intégration correcte
    indices_tri = np.argsort(rappels)
    rappels_tries = rappels[indices_tri]
    precisions_triees = precisions[indices_tri]
    
    # Calculer l'aire sous la courbe avec la règle des trapèzes
    trapezoid = getattr(np, "trapezoid", np.trapz)
    ap = abs(trapezoid(precisions_triees, rappels_tries))
    
    return ap


def pr_especes(
    li_poid: FloatArray,
    Xtest: FloatArray,
    Ytest: FloatArray,
    espece: int = 2,
    step: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Calcule précision et rappel pour une espèce sur différents seuils."""
    scores = np.array([
        neur_score(Xtest[i], li_poid[espece]) for i in range(len(Xtest))
    ])
    seuils = np.arange(scores.min() - 1, scores.max() + 1, step)
    ytrue = Ytest[:, espece].astype(int)
    precisions, rappels = [], []

    for seuil in seuils:
        ypred = (scores > seuil).astype(int)
        vp = np.sum((ytrue == 1) & (ypred == 1))
        fp = np.sum((ytrue == 0) & (ypred == 1))
        fn = np.sum((ytrue == 1) & (ypred == 0))
        rappels.append(vp / (vp + fn) if vp + fn else 0)
        precisions.append(vp / (vp + fp) if vp + fp else 1.0)

    return precisions, rappels
