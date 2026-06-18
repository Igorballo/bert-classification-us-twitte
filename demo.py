"""
demo.py
-------
Interface web interactive (Gradio) pour le modèle BERT fine-tuné.

L'interface :
    - accepte un texte saisi par l'utilisateur (titre + article) ;
    - affiche la classe prédite ET les probabilités de chaque classe ;
    - propose un titre, une description et des exemples pré-remplis.

Lancement :
    python demo.py

Le modèle est chargé depuis ``checkpoints/best_model`` (produit par train.py).

Devoir n°3 - Fine-tuning de BERT pour la classification de texte.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Chemin par défaut du meilleur modèle sauvegardé par train.py.
MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/best_model")

# Variables globales chargées une seule fois au démarrage.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_tokenizer = None
_model = None
_id2label = None
_max_length = 256


def load_artifacts(model_dir: str = MODEL_DIR):
    """Charge le tokenizer, le modèle et la table des labels.

    Args:
        model_dir: Dossier contenant le modèle sauvegardé.
    """
    global _tokenizer, _model, _id2label

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Modèle introuvable dans '{model_dir}'. "
            "Lance d'abord l'entraînement : python train.py"
        )

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(_device)
    _model.eval()

    # Récupération de la correspondance id -> label.
    labels_file = os.path.join(model_dir, "labels.json")
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            data = json.load(f)
        _id2label = {int(k): v for k, v in data["id2label"].items()}
    else:
        _id2label = {int(k): v for k, v in _model.config.id2label.items()}


def predict(text: str) -> dict:
    """Prédit la classe d'un texte et renvoie les probabilités.

    Args:
        text: Le texte (titre + article) saisi par l'utilisateur.

    Returns:
        Dictionnaire {nom_de_classe: probabilité}, format attendu par
        ``gr.Label`` qui affiche un joli classement des probabilités.
    """
    if not text or not text.strip():
        return {"(texte vide)": 1.0}

    encoding = _tokenizer(
        text,
        add_special_tokens=True,
        max_length=_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():  # pas de gradients en inférence
        logits = _model(**encoding).logits
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # On associe chaque probabilité au nom de classe correspondant.
    return {_id2label[i]: float(probs[i]) for i in range(len(probs))}


# Exemples pré-remplis (extraits représentatifs des deux classes).
EXAMPLES = [
    [
        "Senate passes tax reform bill. WASHINGTON (Reuters) - The U.S. Senate "
        "approved a sweeping tax overhaul on Wednesday, handing President Trump "
        "and his Republican allies a major legislative victory before the end "
        "of the year."
    ],
    [
        "North Korea fires missile over Japan. SEOUL (Reuters) - North Korea "
        "launched a ballistic missile that flew over Japan on Friday, the "
        "latest in a series of tests that have raised tensions across the "
        "region and drawn condemnation from world leaders."
    ],
]


def build_interface() -> "gr.Interface":
    """Construit l'interface Gradio.

    Returns:
        L'objet Interface prêt à être lancé.
    """
    description = (
        "Ce modèle BERT a été fine-tuné pour classer un article de presse "
        "en deux catégories : **politicsNews** (politique intérieure US) ou "
        "**worldnews** (actualité internationale). "
        "Collez un titre + article en anglais et obtenez la catégorie prédite "
        "avec les probabilités."
    )

    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=8,
            label="Article (titre + texte, en anglais)",
            placeholder="Collez ici le titre et le corps de l'article...",
        ),
        outputs=gr.Label(num_top_classes=2, label="Catégorie prédite"),
        title="📰 Classification d'articles de presse avec BERT",
        description=description,
        examples=EXAMPLES,
        allow_flagging="never",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démo Gradio")
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--share", action="store_true", help="Lien public Gradio")
    args = parser.parse_args()

    load_artifacts(args.model_dir)
    demo = build_interface()
    demo.launch(share=args.share)
