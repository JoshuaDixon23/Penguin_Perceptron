import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from .metrics import matrice_confusion
from .perceptron import neur, neur_score
def plot_confusion_matrix(conf_matrix):
    """
    Affiche la matrice de confusion.

    Prameters:
    -----------
    conf_matrix : matrice de confusion

    """
    class_names = ['Adélie', 'Gentoo', 'Chinstrap']
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Matrice de confusion')
    fig.colorbar(cax)
    class_ticks = np.arange(len(class_names))
    ax.xaxis.set_major_locator(FixedLocator(class_ticks))
    ax.yaxis.set_major_locator(FixedLocator(class_ticks))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.show()


def plot_roc_espece(specificites, sensibilites, seuil_optimal, x_optimal, y_optimal, auc, espece=0):
    """
    Trace la courbe ROC pour une seule espèce avec la diagonale inverse
    et le point d'intersection optimal.

    Parameters:
    -----------
    specificites : list des spécificités
    sensibilites : list des sensibilités
    seuils : list des seuils
    seuil_optimal : float, seuil optimal
    x_optimal : float, 1 - spécificité au seuil optimal
    y_optimal : float, sensibilité au seuil optimal
    espece : int, 0=Adelie, 1=Gentoo, 2=Chinstrap
    """
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    
    plt.figure(figsize=(7, 7))
    
    # Convertir spécificité en (1 - spécificité) pour l'axe X
    spec_inv = np.array([1 - s for s in specificites])
    sensibilites = np.array(sensibilites)
    
    # Tracer la courbe ROC
    plt.plot(spec_inv, sensibilites, 'b-o', linewidth=2, markersize=4, 
             label=f'{class_names[espece]}')
    
    # Ligne de référence (classifieur aléatoire)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classifieur aléatoire')
    
    # Tracer la diagonale inverse (de haut gauche à bas droit)
    plt.plot([0, 1], [1, 0], 'r--', linewidth=1.5, label='Diagonale inverse')
    
    # Marquer le point optimal
    plt.plot(x_optimal, y_optimal, 'ro', markersize=12, 
             label=f'Seuil validation = {seuil_optimal:.3f}')
    
    # Ajouter une annotation pour le seuil optimal
    plt.annotate(f'Seuil = {seuil_optimal:.3f}\n(1-Spé={x_optimal:.3f}, Sens={y_optimal:.3f})',
                xy=(x_optimal, y_optimal),
                xytext=(x_optimal + 0.15, y_optimal - 0.15),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))
    
    
    plt.title(f'Courbe ROC - {class_names[espece]}\nAUC = {auc:.3f}', fontsize=14)
    plt.xlabel('Taux de faux positifs (1 - Spécificité)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (Sensibilité)', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"Résultats pour {class_names[espece]}:")
    print(f"{'='*60}")
    print(f"AUC (Aire sous la courbe): {auc:.3f}")
    print(f"\nSeuil optimal (intersection avec diagonale inverse): {seuil_optimal:.3f}")
    print(f"  - Sensibilité: {y_optimal:.3f}")
    print(f"  - 1-Spécificité: {x_optimal:.3f}")
    print(f"{'='*60}")
    
    if auc > 0.9:
        print("→ Excellent classifieur !")
    elif auc > 0.7:
        print("→ Bon classifieur")
    elif auc > 0.5:
        print("→ Classifieur faible")
    else:
        print("→ Classifieur très mauvais (pire que le hasard)")
    
    return seuil_optimal


def plot_roc_espece_bruit(specificites, specificite_bruit, sensibilites, sensibilites_bruit, seuil_optimal, seuil_opti_bruit, x_optimal, x_optimal_bruit, y_optimal, y_optimal_bruit, auc, auc_bruit,espece=0):
    """
    Trace la courbe ROC pour une seule espèce avec la diagonale inverse
    et le point d'intersection optimal.

    Parameters:
    -----------
    specificites : list des spécificités
    specificite_bruit : list des spécificités avec bruit
    sensibilites : list des sensibilités
    sensibilites_bruit : list des sensibilités avec bruit
    seuil_optimal : float, seuil optimal
    seuil_opti_bruit : float, seuil optimal avec bruit
    x_optimal : float, 1 - spécificité au seuil optimal
    y_optimal : float, sensibilité au seuil optimal
    espece : int, 0=Adelie, 1=Gentoo, 2=Chinstrap
    """
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    
    plt.figure(figsize=(5, 5))
    
    # Convertir spécificité en (1 - spécificité) pour l'axe X
    spec_inv = np.array([1 - s for s in specificites])
    spec_inv_bruit = np.array([1 - s for s in specificite_bruit])
    sensibilites = np.array(sensibilites)
    sensibilites_bruit = np.array(sensibilites_bruit)
    
    # Tracer la courbe ROC
    plt.plot(spec_inv, sensibilites, 'b-o', linewidth=2, markersize=4, 
             label=f'{class_names[espece]}')
    
    plt.plot(spec_inv_bruit, sensibilites_bruit, 'g--', linewidth=2, markersize=4,
             label=f'{class_names[espece]} avec bruit')
    
    # Ligne de référence (classifieur aléatoire)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classifieur aléatoire')
    
    # Tracer la diagonale inverse (de haut gauche à bas droit)
    plt.plot([0, 1], [1, 0], 'r--', linewidth=1.5, label='Diagonale inverse')
    
    # Marquer le point optimal
    plt.plot(x_optimal, y_optimal, 'ro', markersize=12, 
             label=f'Seuil optimal = {seuil_optimal:.3f}')
    
    plt.plot(x_optimal_bruit, y_optimal_bruit, 'mo', markersize=12,
             label=f'Même seuil sur données bruitées = {seuil_opti_bruit:.3f}')    
    
    plt.title(f'Courbe ROC - {class_names[espece]}\nAUC = {auc:.3f}', fontsize=14)
    plt.xlabel('Taux de faux positifs (1 - Spécificité)', fontsize=12)
    plt.ylabel('Taux de vrais positifs (Sensibilité)', fontsize=12)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"Résultats pour {class_names[espece]}:")
    print(f"{'='*60}")
    print(f"AUC (Aire sous la courbe): {auc:.3f}")
    print(f"\nSeuil optimal (intersection avec diagonale inverse): {seuil_optimal:.3f}")
    print(f"  - Sensibilité: {y_optimal:.3f}")
    print(f"  - 1-Spécificité: {x_optimal:.3f}")
    print(f"{'='*60}")
    
    if auc > 0.9:
        print("→ Excellent classifieur !")
    elif auc > 0.7:
        print("→ Bon classifieur")
    elif auc > 0.5:
        print("→ Classifieur faible")
    else:
        print("→ Classifieur très mauvais (pire que le hasard)")
    
    return seuil_optimal


def test_avec_seuils_optimaux(Xtest, Ytest, li_poid, seuils_optimaux):
    """
    Teste le modèle avec des seuils optimaux différents pour chaque espèce.
    
    Parameters:
    -----------
    seuils_optimaux : list [seuil_adelie, seuil_gentoo, seuil_chinstrap]
    """
    Ypred = np.zeros_like(Ytest)
    
    for i in range(len(Xtest)):
        li_e = [Xtest[i][0], Xtest[i][1], Xtest[i][2], Xtest[i][3]]
        
        # Calculer les scores pour chaque espèce avec son seuil optimal
        for j in range(len(li_poid)):
            score = neur_score(li_e, li_poid[j])
            # Utiliser le seuil optimal de cette espèce
            Ypred[i][j] = 1 if score > seuils_optimaux[j] else 0
    
    conf_matrix = matrice_confusion(Ytest, Ypred)
    return conf_matrix


def afficher_scores_manchots(Xtest, Ytest, li_poid, classe_cible=0):
    """
    Affiche un tableau avec les manchots, leur vrai label et leur score
    pour une classe donnée.
    
    Parameters:
    -----------
    Xtest : array des données de test
    Ytest : array des vrais labels
    li_poid : poids du modèle entraîné
    classe_cible : 0=Adelie, 1=Gentoo, 2=Chinstrap
    """
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    
    # Calculer les scores pour tous les manchots
    scores = []
    vrais_labels = []
    
    for i in range(len(Xtest)):
        li_e = [Xtest[i][0], Xtest[i][1], Xtest[i][2], Xtest[i][3]]
        score = neur_score(li_e, li_poid[classe_cible])
        scores.append(score)
        
        # Trouver le vrai label
        vrai_classe = np.argmax(Ytest[i])
        vrais_labels.append(class_names[vrai_classe])
    
    # Créer un DataFrame pour un affichage propre
    df_scores = pd.DataFrame({
        'Manchot': [f"M{i+1}" for i in range(len(Xtest))],
        'Vrai label': vrais_labels,
        'Score': scores
    })
    
    # Trier par score
    df_scores = df_scores.sort_values('Score').reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Scores pour la classe: {class_names[classe_cible]}")
    print(f"{'='*60}")
    print(df_scores.to_string(index=False))
    print(f"{'='*60}")
    
    # Statistiques
    print(f"\nStatistiques:")
    print(f"  Score min: {min(scores):.2f}")
    print(f"  Score max: {max(scores):.2f}")
    print(f"  Score moyen: {np.mean(scores):.2f}")
    print(f"  Écart-type: {np.std(scores):.2f}")
    
    # Compter les vrais positifs et négatifs
    est_classe_cible = [1 if label == class_names[classe_cible] else 0 for label in vrais_labels]
    print(f"\n  Nombre de vrais {class_names[classe_cible]}: {sum(est_classe_cible)}")
    print(f"  Nombre de non-{class_names[classe_cible]}: {len(est_classe_cible) - sum(est_classe_cible)}")


def plot_pr_espece(precisions, rappels, ap, espece=0):
    """
    Trace la courbe Précision-Rappel pour une espèce donnée.
    
    Parameters:
    -----------
    precisions : list des précisions
    rappels : list des rappels
    espece : int, 0=Adelie, 1=Gentoo, 2=Chinstrap
    """
    class_names = ["Adelie", "Gentoo", "Chinstrap"]
    
    plt.figure(figsize=(8, 8))
    
    precisions = np.array(precisions)
    rappels = np.array(rappels)
    
    # CORRECTION : Trier par rappel croissant pour avoir une courbe propre
    indices_tri = np.argsort(rappels)
    rappels_tries = rappels[indices_tri]
    precisions_triees = precisions[indices_tri]
    
    # Tracer la courbe Précision-Rappel
    plt.plot(rappels_tries, precisions_triees, 'b-', linewidth=2, 
             label=f'{class_names[espece]}')
    
    
    plt.title(f'Courbe Précision-Rappel - {class_names[espece]}\nAP = {ap:.3f}', 
              fontsize=14)
    plt.xlabel('Rappel (Sensibilité)', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.legend(fontsize=11, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"Average Precision (AP) pour {class_names[espece]}: {ap:.3f}")
    print(f"{'='*60}")
