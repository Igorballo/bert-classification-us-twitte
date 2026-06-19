"""
train.py
--------
Boucle d'entraînement PyTorch PURE pour le fine-tuning de BERT.

Conformément à l'énoncé, le Trainer de Hugging Face n'est PAS utilisé :
les fonctions ``train_epoch`` et ``eval_epoch`` sont écrites manuellement.

Pipeline :
    1. Fixation de la graine (reproductibilité).
    2. Chargement du CSV, split train/validation 80/20 stratifié.
    3. Tokenizer + modèle BERT pré-entraîné.
    4. Optimiseur AdamW + scheduler linéaire avec warmup.
    5. Boucle d'epochs : entraînement puis validation.
    6. Collecte des métriques (loss, accuracy, F1) à chaque epoch.
    7. Sauvegarde du MEILLEUR modèle (meilleure val_loss).
    8. Visualisations (courbes + matrice de confusion) à la fin.

Lancement :
    python train.py --csv data/True.csv --epochs 3 --batch_size 16

Devoir n°3 - Fine-tuning de BERT pour la classification de texte.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import (
    load_dataframe,
    split_data,
    TextClassificationDataset,
    CLASS_NAMES,
    ID2LABEL,
    LABEL2ID,
)
from model import load_tokenizer, load_model, DEFAULT_MODEL_NAME
from utils import set_seed, compute_metrics, plot_history, plot_confusion_matrix


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Entraîne le modèle sur une epoch complète.

    Args:
        model: Le modèle BERT.
        dataloader: DataLoader d'entraînement.
        optimizer: L'optimiseur (AdamW).
        scheduler: Le scheduler de learning rate (peut être None).
        criterion: La fonction de perte (CrossEntropyLoss).
        device: 'cuda' ou 'cpu'.

    Returns:
        Tuple (loss moyenne, accuracy, f1) sur l'epoch.
    """
    model.train()  # mode entraînement (active dropout, etc.)
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # remise à zéro des gradients

        # Passe avant : on passe le masque d'attention au modèle.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Passe arrière + mise à jour des poids.
        loss.backward()
        # Clipping du gradient : stabilise le fine-tuning de BERT.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1"]


@torch.no_grad()  # désactive le calcul des gradients pendant la validation
def eval_epoch(model, dataloader, criterion, device):
    """Évalue le modèle sur l'ensemble de validation.

    Le décorateur ``@torch.no_grad()`` et ``model.eval()`` désactivent
    respectivement le calcul des gradients et le dropout : indispensables
    pour une validation correcte et économe en mémoire.

    Args:
        model: Le modèle BERT.
        dataloader: DataLoader de validation.
        criterion: La fonction de perte (CrossEntropyLoss).
        device: 'cuda' ou 'cpu'.

    Returns:
        Tuple (loss moyenne, accuracy, f1, y_true, y_pred).
    """
    model.eval()  # mode évaluation
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1"], all_labels, all_preds


def main():
    """Point d'entrée : orchestre tout l'entraînement."""
    parser = argparse.ArgumentParser(description="Fine-tuning de BERT")
    parser.add_argument("--csv", default="data/True.csv", help="Chemin du CSV")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limite le nb d'exemples (utile pour un test rapide).",
    )
    args = parser.parse_args()

    # --- 1. Reproductibilité ---
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device : {device}")

    # --- 2. Données ---
    df = load_dataframe(args.csv)
    if args.max_samples is not None:
        # Échantillon stratifié rapide pour tester le pipeline.
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.max_samples // 2), random_state=args.seed))
            .reset_index(drop=True)
        )
    print(f"[train] Exemples utilisés : {len(df)}")

    train_df, val_df = split_data(df, test_size=0.2, seed=args.seed)
    print(f"[train] Train : {len(train_df)} | Validation : {len(val_df)}")

    # --- 3. Tokenizer + modèle ---
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_ds = TextClassificationDataset(
        train_df["input_text"], train_df["label"], tokenizer, args.max_length
    )
    val_ds = TextClassificationDataset(
        val_df["input_text"], val_df["label"], tokenizer, args.max_length
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # --- 4. Optimiseur + scheduler + loss ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()  # attend des labels entiers

    # --- 5/6. Boucle d'entraînement + collecte des métriques ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_f1_score": [], "learning_rate": [],
    }
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best_model")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        va_loss, va_acc, va_f1, y_true, y_pred = eval_epoch(
            model, val_loader, criterion, device
        )

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_accuracy"].append(tr_acc)
        history["val_accuracy"].append(va_acc)
        history["val_f1_score"].append(va_f1)
        history["learning_rate"].append(current_lr)

        print(
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # --- 7. Sauvegarde du meilleur modèle (meilleure val_loss) ---
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            with open(os.path.join(best_path, "labels.json"), "w") as f:
                json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)
            print(f"[train] ✅ Meilleur modèle sauvegardé (val_loss={va_loss:.4f})")
            best_y_true, best_y_pred = y_true, y_pred

    # Sauvegarde de l'historique des métriques (pour le rapport).
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # --- 8. Visualisations ---
    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))
    plot_confusion_matrix(
        best_y_true, best_y_pred, CLASS_NAMES,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    print(f"\n[train] Terminé. Meilleure val_loss = {best_val_loss:.4f}")
    print(f"[train] Modèle disponible dans : {best_path}")


if __name__ == "__main__":
    main()
