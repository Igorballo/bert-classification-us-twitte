"""
utils.py
--------
Fonctions utilitaires partagées par le projet :
    - reproductibilité (fixation des graines aléatoires) ;
    - calcul des métriques de classification (accuracy, F1) ;
    - visualisation des courbes d'apprentissage et de la matrice de confusion.

Devoir n°3 - Fine-tuning de BERT pour la classification de texte.
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def set_seed(seed: int = 42) -> None:
    """Fixe toutes les graines aléatoires pour garantir la reproductibilité.

    On fixe les graines de ``random``, ``numpy`` et ``torch`` (CPU + CUDA),
    puis on force cuDNN en mode déterministe.

    Args:
        seed: Valeur de la graine à utiliser.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Comportement déterministe de cuDNN (au prix d'un léger surcoût).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Variable d'environnement utilisée par certaines opérations CUDA.
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_metrics(y_true, y_pred) -> dict:
    """Calcule les métriques de classification.

    Args:
        y_true: Labels réels (liste ou tableau d'entiers).
        y_pred: Labels prédits (liste ou tableau d'entiers).

    Returns:
        Dictionnaire contenant ``accuracy`` et ``f1`` (F1 pondéré, adapté
        même en cas de léger déséquilibre des classes).
    """
    accuracy = accuracy_score(y_true, y_pred)
    # F1 "weighted" : moyenne des F1 par classe pondérée par le support.
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


def plot_history(history: dict, out_path: str = "training_curves.png") -> None:
    """Trace les courbes loss et accuracy (train vs validation).

    Args:
        history: Dictionnaire produit par ``train.py`` contenant les listes
            ``train_loss``, ``val_loss``, ``train_accuracy``, ``val_accuracy``.
        out_path: Chemin de sauvegarde de l'image.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Courbe de perte ---
    axes[0].plot(epochs, history["train_loss"], "o-", label="train_loss")
    axes[0].plot(epochs, history["val_loss"], "o-", label="val_loss")
    axes[0].set_title("Loss par epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Courbe d'accuracy ---
    axes[1].plot(epochs, history["train_accuracy"], "o-", label="train_accuracy")
    axes[1].plot(epochs, history["val_accuracy"], "o-", label="val_accuracy")
    axes[1].set_title("Accuracy par epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Courbes d'apprentissage sauvegardées dans : {out_path}")


def plot_confusion_matrix(
    y_true, y_pred, class_names, out_path: str = "confusion_matrix.png"
) -> None:
    """Trace et sauvegarde la matrice de confusion.

    Args:
        y_true: Labels réels.
        y_pred: Labels prédits.
        class_names: Liste des noms de classes (ordre = index du label).
        out_path: Chemin de sauvegarde de l'image.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion (validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[utils] Matrice de confusion sauvegardée dans : {out_path}")

    # Affichage texte du rapport détaillé (précision/rappel/F1 par classe).
    print("\n[utils] Rapport de classification :\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
