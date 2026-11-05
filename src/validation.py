from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .data import get_sets
from .metrics import calculer_auc, compute_mauc, roc_espece
from .perceptron import neur

FloatArray = NDArray[np.float64]

def t_etoile(k, lr, etapes, seuil=0, donnees=None, seed=42):
    """
    Apprentissage d'un perceptron multi-classes avec validation croisée k-fold et suivi du MAUC.
    
    Parameters:
    -----------
    k : nombre de folds pour la validation croisée k-fold
    lr : taux d'apprentissage
    etapes : nombre d'étapes d'apprentissage
    seuil : seuil d'activation
    seed : graine utilisée pour les séparations et l'initialisation des poids

    Returns:
    --------
    moyenne_mauc_app_par_etape : moyenne des MAUC pour chaque étape (apprentissage)
    moyenne_mauc_val_par_etape : moyenne des MAUC pour chaque étape (validation)
    """
    mat_mauc_app = []
    mat_mauc_val = []
    
    # Récupérer les données une seule fois
    if donnees is None:
        print("Erreur : les données doivent être fournies à la fonction t_etoile.")
        return None, None
    entrees_modele, sorties_modele = donnees
    Xapp, Yapp, Xvalid, Yvalid, _, _ = get_sets(
        entrees_modele, sorties_modele, 60, 20, random_state=seed
    )
    
    # Créer les k folds
    folds = get_li_k_fold(Xapp, Yapp, Xvalid, Yvalid, k, seed=seed)
    rng = np.random.default_rng(seed)
    
    # Itérer sur chaque fold
    for fold_idx in range(k):
        Xapp_fold, Yapp_fold, Xvalid_fold, Yvalid_fold = folds[fold_idx]

        
        li_mauc_app_etape = []
        li_mauc_val_etape = []
        
        nb_neur = 3
        li_poid = rng.random((3, 4))
        
        for etape in range(etapes):
            # Apprentissage
            for i in range(len(Xapp_fold)):
                li_e = [Xapp_fold[i][0], Xapp_fold[i][1], Xapp_fold[i][2], Xapp_fold[i][3]]
                for j in range(nb_neur):
                    nr_poid = li_poid[j]
                    s = neur(li_e, nr_poid, seuil)
                    for l in range(len(nr_poid)):
                        y = Yapp_fold[i][j]
                        e = (y - s)
                        if y != s:
                            nr_poid[l] = nr_poid[l] + (lr * e * li_e[l])
            
            # Calcul MAUC sur les données d'apprentissage du fold
            mauc_app_etape = 0
            for j in range(nb_neur):
                sp, se, _ = roc_espece(li_poid, Xapp_fold, Yapp_fold, espece=j, step=0.01)
                auc = calculer_auc(sp, se)
                mauc_app_etape += auc
            mauc_app_etape /= nb_neur
            li_mauc_app_etape.append(mauc_app_etape)

            # Calcul MAUC sur les données de validation du fold
            mauc_val_etape = 0
            for j in range(nb_neur):
                sp, se, _ = roc_espece(li_poid, Xvalid_fold, Yvalid_fold, espece=j, step=0.01)
                auc = calculer_auc(sp, se)
                mauc_val_etape += auc
            mauc_val_etape /= nb_neur
            li_mauc_val_etape.append(mauc_val_etape)
        
        mat_mauc_app.append(li_mauc_app_etape)
        mat_mauc_val.append(li_mauc_val_etape)
    
    # Moyenne sur tous les folds
    moyenne_mauc_app_par_etape = np.mean(mat_mauc_app, axis=0)
    moyenne_mauc_val_par_etape = np.mean(mat_mauc_val, axis=0)
    
    return moyenne_mauc_app_par_etape, moyenne_mauc_val_par_etape


def plot_t_etoile(moyenne_mauc_app, moyenne_mauc_val):
    """
    Trace les courbes de MAUC (Mean AUC) pour l'apprentissage et la validation.
    
    Parameters:
    -----------
    moyenne_mauc_app : array des moyennes de MAUC pour chaque étape (apprentissage)
    moyenne_mauc_val : array des moyennes de MAUC pour chaque étape (validation)
    """
    etapes = np.arange(len(moyenne_mauc_app))

    find_t_etoile = np.argmax(moyenne_mauc_val)
    print(f"\nÉtape optimale (t_etoile) basée sur la validation : {find_t_etoile}")

    
    plt.figure(figsize=(12, 6))
    
    # Tracer les courbes
    plt.plot(etapes, moyenne_mauc_app, 'b-o', linewidth=2, markersize=5, 
             label='MAUC Apprentissage', alpha=0.8)
    plt.plot(etapes, moyenne_mauc_val, 'r-s', linewidth=2, markersize=5, 
             label='MAUC Validation', alpha=0.8)
    
    plt.axvline(x=find_t_etoile, color='g', linestyle='--', label=f"t_etoile = {find_t_etoile}")
    
    plt.title('Évolution du MAUC (Mean AUC) par étape', fontsize=14, fontweight='bold')
    plt.xlabel('Étapes d\'apprentissage', fontsize=12)
    plt.ylabel('MAUC', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques
    print(f"\n{'='*60}")
    print(f"Statistiques du MAUC")
    print(f"{'='*60}")
    print(f"\nApprentissage:")
    print(f"  MAUC initial: {moyenne_mauc_app[0]:.4f}")
    print(f"  MAUC final: {moyenne_mauc_app[-1]:.4f}")
    print(f"  MAUC moyen: {np.mean(moyenne_mauc_app):.4f}")
    print(f"  MAUC max: {np.max(moyenne_mauc_app):.4f} (étape {np.argmax(moyenne_mauc_app)})")
    print(f"  MAUC min: {np.min(moyenne_mauc_app):.4f} (étape {np.argmin(moyenne_mauc_app)})")
    
    print(f"\nValidation:")
    print(f"  MAUC initial: {moyenne_mauc_val[0]:.4f}")
    print(f"  MAUC final: {moyenne_mauc_val[-1]:.4f}")
    print(f"  MAUC moyen: {np.mean(moyenne_mauc_val):.4f}")
    print(f"  MAUC max: {np.max(moyenne_mauc_val):.4f} (étape {np.argmax(moyenne_mauc_val)})")
    print(f"  MAUC min: {np.min(moyenne_mauc_val):.4f} (étape {np.argmin(moyenne_mauc_val)})")
    print(f"{'='*60}")


def applique_bruit(
    t: FloatArray,
    b: float,
    seed: int | None = None,
) -> FloatArray:
    tab = np.copy(t) 
    rng = np.random.default_rng(seed)
    for i in range(len(tab)):
        sca1, sca2, sca3, sca4 = rng.uniform(-b, b, size=4)
        tab[i][0] *= (1 + sca1)
        tab[i][1] *= (1 + sca2)
        tab[i][2] *= (1 + sca3)
        tab[i][3] *= (1 + sca4)
    return tab


def find_k_etoile(
    nb_folds_max=10,
    seed=42,
    lr=.001,
    etapes=300,
    seuil=0,
    donnees=None,
):
    """
    Trouve le meilleur k pour la validation croisée k-fold en fonction de l'erreur du MAUC moyen.
    L'erreur est calculée comme la différence absolue entre MAUC apprentissage et validation.
    Un bon k devrait minimiser cette erreur (pas la maximiser).
    
    Parameters:
    -----------
    donnees : tuple (entrees, sorties), les données à utiliser pour la validation croisée
    nb_folds_max : int, nombre maximum de folds à tester
    seed : int, graine aléatoire pour la reproductibilité
    lr : float, taux d'apprentissage (ajouté pour la complétude)
    etapes : int, nombre d'étapes (époques) d'entraînement (ajouté pour la complétude)
    seuil : float, seuil d'activation (ajouté pour la complétude)
    
    Returns:
    --------
    best_k : int, meilleur k (celui avec la plus petite erreur)
    li_erreur_k : list, liste des erreurs moyennes pour chaque k
    """
    if donnees is None:
            print("Erreur : les données doivent être fournies à la fonction find_k_etoile.")
            return None, None
    X, Y = np.array(donnees[0]), np.array(donnees[1])
    rng = np.random.default_rng(seed)
    
    best_k = 2
    best_ecart = float('inf')
    li_k = list(range(2, nb_folds_max + 1))
    li_erreur_k = []

    for k in li_k:
        folds = liste_k_fold(X, Y, k, seed=seed)
        ecarts_folds = []
        
        for fold_idx in range(k):
            (Xapp_fold, Yapp_fold), (Xvalid_fold, Yvalid_fold) = folds[fold_idx]
            Xapp_np, Yapp_np = np.array(Xapp_fold), np.array(Yapp_fold)
            Xvalid_np, Yvalid_np = np.array(Xvalid_fold), np.array(Yvalid_fold)

            moy = Xapp_np.mean(axis=0)
            std = Xapp_np.std(axis=0)
            std[std == 0] = 1
            Xapp_np = (Xapp_np - moy) / std
            Xvalid_np = (Xvalid_np - moy) / std
            
            li_poid = rng.random((3, 4))
            
            # Boucle d'entraînement restaurée
            for _ in range(etapes):
                for i in range(len(Xapp_np)):
                    li_e = Xapp_np[i]
                    for j in range(3):
                        s = neur(li_e, li_poid[j], seuil)
                        y = Yapp_np[i][j]
                        if y != s:
                            li_poid[j] += lr * (y - s) * li_e

            mauc_app = compute_mauc(Xapp_np, Yapp_np, li_poid)
            mauc_valid = compute_mauc(Xvalid_np, Yvalid_np, li_poid)
            ecarts_folds.append(abs(mauc_app - mauc_valid))

        mean_ecart = np.mean(ecarts_folds)
        li_erreur_k.append(mean_ecart)

        if mean_ecart < best_ecart:
            best_ecart = mean_ecart
            best_k = k

    return best_k, li_erreur_k, li_k


def plot_find_k_etoile(k_etoile,li_erreur , li_k):
    """
    Trace lErreur moyenne en fonction de k pour la validation croisée k-fold.
    
    Parameters:
    -----------
    li_mauc_etape : list des MAUC moyens pour chaque k
    """

    plt.figure(figsize=(10, 5))
    plt.plot(li_k, li_erreur, 'b-o', linewidth=2, markersize=6)
    plt.axvline(x=k_etoile, color='g', linestyle='--', label=f"k_etoile = {k_etoile}")
    plt.title('Erreur moyenne en fonction de k (k-fold)', fontsize=14)
    plt.xlabel('k (nombre de folds)', fontsize=12)
    plt.ylabel('Écart absolu moyen de MAUC', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def liste_k_fold(X, Y, k, seed=42):
    """
    Génère les paires (train_set, valid_set) pour la validation croisée k-fold.

    Args:
        X (list ou array-like): La liste ou le tableau des données d'entrée.
        Y (list ou array-like): La liste ou le tableau des étiquettes (cibles).
        k (int): Le nombre de plis (folds) à utiliser.

    Returns:
        list: Une liste de k éléments, où chaque élément est un tuple
              contenant deux listes : (train_set, valid_set).
    """
    # Convertir en tableau NumPy pour une indexation facile si ce n'est pas déjà le cas
    X = np.array(X)
    Y = np.array(Y)

    from sklearn.model_selection import KFold
    # Initialiser KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Liste pour stocker les résultats
    k_fold_sets = []

    # Parcourir les indices générés par KFold
    for train_index, valid_index in kf.split(X):
        # Créer les ensembles d'entraînement
        X_train, X_valid = X[train_index], X[valid_index]
        Y_train, Y_valid = Y[train_index], Y[valid_index]
        
        # Combiner X et Y pour former le train set et le valid set
        # Nous renvoyons généralement les données X et les étiquettes Y séparément
        # dans les listes pour chaque ensemble.
        
        train_set = [X_train.tolist(), Y_train.tolist()]
        valid_set = [X_valid.tolist(), Y_valid.tolist()]
        
        k_fold_sets.append((train_set, valid_set))

    return k_fold_sets


def get_li_k_fold(Xapp, Yapp, Xvalid, Yvalid, k, seed=42):
    """
    Divise les données en k folds pour la validation croisée.   

    Parameters:
    -----------
    Xapp : array des données d'apprentissage
    Yapp : array des labels d'apprentissage
    Xvalid : array des données de validation
    Yvalid : array des labels de validation
    k : int, nombre de folds

    Returns:
    --------
    folds : list de tuples (X_fold, Y_fold) pour chaque fold
    """
    X_full = np.vstack((Xapp, Xvalid))
    Y_full = np.vstack((Yapp, Yvalid))
    
    indices = np.arange(len(X_full))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds_idx = np.array_split(indices, k)
    
    folds = []
    for i in range(k):
        val_idx = folds_idx[i]
        train_idx = np.hstack([folds_idx[j] for j in range(k) if j != i])
        
        folds.append((
            X_full[train_idx], Y_full[train_idx],
            X_full[val_idx], Y_full[val_idx]
        ))
    return folds
