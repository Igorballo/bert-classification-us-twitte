"""
dataset.py
----------
Chargement des données et classe ``Dataset`` PyTorch personnalisée.

Tâche : classifier la concaténation ``title + text`` d'un article de presse
selon la colonne ``subject`` (``politicsNews`` vs ``worldnews``).

Le fichier expose :
    - LABEL2ID / ID2LABEL : la correspondance label <-> entier ;
    - load_dataframe()    : lecture du CSV + nettoyage minimal ;
    - split_data()        : split train/validation 80/20 STRATIFIÉ ;
    - TextClassificationDataset : Dataset PyTorch (tokenization à la volée).

Devoir n°3 - Fine-tuning de BERT pour la classification de texte.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Correspondance explicite label texte <-> entier.
# IMPORTANT : la CrossEntropyLoss attend des labels ENTIERS (pas du one-hot).
LABEL2ID = {"politicsNews": 0, "worldnews": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES = [ID2LABEL[i] for i in range(len(ID2LABEL))]


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Charge le CSV et construit les colonnes ``input_text`` et ``label``.

    Le texte d'entrée est la concaténation du titre et du corps de l'article :
    ``title + " " + text``. Les lignes dont le ``subject`` n'est pas connu
    sont écartées, ainsi que les textes vides.

    Args:
        csv_path: Chemin vers le fichier CSV (colonnes title, text, subject, date).

    Returns:
        DataFrame avec deux colonnes : ``input_text`` (str) et ``label`` (int).
    """
    df = pd.read_csv(csv_path)

    # On ne garde que les sujets attendus (robustesse si d'autres apparaissent).
    df = df[df["subject"].isin(LABEL2ID)].copy()

    # Concaténation titre + texte ; on remplace les valeurs manquantes par "".
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    df["input_text"] = (title + " " + text).str.strip()

    # Suppression des textes vides après nettoyage.
    df = df[df["input_text"].str.len() > 0].copy()

    # Conversion du label texte en entier.
    df["label"] = df["subject"].map(LABEL2ID).astype(int)

    return df[["input_text", "label"]].reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Découpe le DataFrame en train/validation de façon STRATIFIÉE.

    La stratification sur ``label`` garantit que la proportion des deux classes
    est identique dans le train et la validation.

    Args:
        df: DataFrame issu de ``load_dataframe``.
        test_size: Proportion réservée à la validation (0.2 = 20 %).
        seed: Graine pour la reproductibilité du split.

    Returns:
        Tuple (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],  # split stratifié
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class TextClassificationDataset(Dataset):
    """Dataset PyTorch qui tokenize les textes à la volée.

    Chaque élément renvoie un dictionnaire contenant :
        - ``input_ids``      : identifiants des tokens (tenseur long) ;
        - ``attention_mask`` : masque d'attention (1 = token réel, 0 = padding) ;
        - ``labels``         : le label entier (tenseur long).

    Le masque d'attention est ESSENTIEL : l'oublier dégrade fortement les
    résultats car le modèle "regarde" alors les tokens de padding.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        """
        Args:
            texts: Itérable de chaînes de caractères (les textes d'entrée).
            labels: Itérable d'entiers (les labels correspondants).
            tokenizer: Tokenizer Hugging Face (ex. BertTokenizerFast).
            max_length: Longueur maximale de séquence (troncature/padding).
        """
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Nombre d'exemples dans le dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Tokenize l'exemple ``idx`` et renvoie les tenseurs prêts pour BERT."""
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # ajoute [CLS] et [SEP]
            max_length=self.max_length,
            padding="max_length",         # padding jusqu'à max_length
            truncation=True,              # troncature des textes trop longs
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Le tokenizer renvoie un batch de taille 1 -> on enlève la 1re dim.
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Test rapide du module exécuté seul : python dataset.py
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test du module dataset.py")
    parser.add_argument("--csv", default="data/True.csv", help="Chemin du CSV")
    args = parser.parse_args()

    frame = load_dataframe(args.csv)
    print(f"Nombre total d'exemples : {len(frame)}")
    print("Distribution des classes :")
    print(frame["label"].map(ID2LABEL).value_counts())
    print("\n5 exemples :")
    for i in range(5):
        txt = frame.loc[i, "input_text"]
        print(f"  [{ID2LABEL[frame.loc[i, 'label']]}] {txt[:120]}...")

    tr, va = split_data(frame)
    print(f"\nTrain : {len(tr)} | Validation : {len(va)}")
