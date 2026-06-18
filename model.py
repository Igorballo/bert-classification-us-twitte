"""
model.py
--------
Chargement du tokenizer et du modèle BERT pré-entraîné.

On utilise ``bert-base-uncased`` (dataset en anglais) avec une tête de
classification linéaire à ``num_labels`` sorties, fournie par la classe
``BertForSequenceClassification`` de Hugging Face.

NB : on charge UNIQUEMENT le modèle pré-entraîné et son tokenizer ;
toute la boucle d'entraînement est écrite manuellement dans ``train.py``
(le Trainer de Hugging Face n'est PAS utilisé, conformément à l'énoncé).

Devoir n°3 - Fine-tuning de BERT pour la classification de texte.
"""

# Rend la syntaxe d'annotation moderne (ex. ``dict | None``) compatible
# avec Python 3.9 : les annotations ne sont plus évaluées à l'exécution.
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_NAME = "bert-base-uncased"


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Charge le tokenizer associé au modèle pré-entraîné.

    Args:
        model_name: Nom du modèle Hugging Face.

    Returns:
        Un tokenizer (version "fast" si disponible).
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 2,
    id2label: dict | None = None,
    label2id: dict | None = None,
):
    """Charge le modèle BERT pré-entraîné avec une tête de classification.

    La tête de classification (couche linéaire au-dessus du token [CLS]) est
    initialisée aléatoirement : c'est elle, ainsi que les poids de BERT, qui
    seront affinés (fine-tuning) pendant l'entraînement.

    Args:
        model_name: Nom du modèle Hugging Face pré-entraîné.
        num_labels: Nombre de classes de sortie.
        id2label: Correspondance index -> nom de classe (pour la lisibilité).
        label2id: Correspondance nom de classe -> index.

    Returns:
        Une instance de ``BertForSequenceClassification``.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return model


if __name__ == "__main__":
    # Test rapide : vérifie le chargement et compte les paramètres.
    tok = load_tokenizer()
    mdl = load_model()
    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"Tokenizer chargé : {tok.__class__.__name__}")
    print(f"Modèle chargé    : {mdl.__class__.__name__}")
    print(f"Nombre de paramètres : {n_params / 1e6:.1f} M")
