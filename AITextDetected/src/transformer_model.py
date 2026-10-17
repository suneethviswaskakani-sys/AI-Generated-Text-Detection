"""
Model D (optional): transformer-based text classifier.

Design goals, per project spec:
- Optional. If `torch`/`transformers` are missing, or the saved model
  directory doesn't exist, every caller must be able to detect that and
  skip this model without crashing the app.
- Local, no paid API. Default model is `distilbert-base-uncased`
  (~66M params) specifically because it is small enough to fine-tune and
  run on a CPU-only student laptop, per the project's laptop-friendliness
  requirement -- RoBERTa/DeBERTa-v3-small are drop-in alternatives via
  config.TRANSFORMER_MODEL_NAME if more compute is available.

HONESTY NOTE (read this before trusting any number attached to this file):
This module was written and reviewed in a sandboxed environment with NO
internet access and neither `torch` nor `transformers` installed, so it
could not actually be executed here -- no fine-tuning run, no accuracy
number, nothing. Do not report metrics for this model unless you have
actually run `training/train_transformer.py` yourself and read the numbers
out of the resulting `data/reports/transformer_metrics.json`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config


class TransformerUnavailableError(RuntimeError):
    """Raised when torch/transformers aren't installed, or no trained model exists."""
    pass


def _require_libs():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise TransformerUnavailableError(
            "torch and/or transformers is not installed. Install with "
            "`pip install torch transformers` (requires internet access) "
            "to use the transformer model. The rest of the app works without it."
        ) from e


def is_available() -> bool:
    """Cheap check: are the libraries importable? (Does not check for saved weights.)"""
    try:
        _require_libs()
        return True
    except TransformerUnavailableError:
        return False


class TransformerTextClassifier:
    """Thin wrapper around a HuggingFace sequence-classification model."""

    def __init__(self, model_name: str = None, max_length: int = None):
        _require_libs()
        self.model_name = model_name or config.TRANSFORMER_MODEL_NAME
        self.max_length = max_length or config.TRANSFORMER_MAX_LENGTH
        self.tokenizer = None
        self.model = None

    def _init_fresh(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

    def fit(self, train_texts, train_labels, valid_texts, valid_labels,
            epochs: int = None, batch_size: int = None, lr: float = None):
        """
        Fine-tune on (train_texts, train_labels); evaluate each epoch on
        (valid_texts, valid_labels). Uses plain PyTorch (no Trainer
        dependency) so behavior is transparent and doesn't require
        `accelerate`. Runs on GPU automatically if `torch.cuda.is_available()`.
        """
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW

        if self.model is None:
            self._init_fresh()

        epochs = epochs or config.TRANSFORMER_EPOCHS
        batch_size = batch_size or config.TRANSFORMER_BATCH_SIZE
        lr = lr or config.TRANSFORMER_LR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = list(texts)
                self.labels = list(labels)
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx], truncation=True, padding="max_length",
                    max_length=self.max_length, return_tensors="pt",
                )
                item = {k: v.squeeze(0) for k, v in enc.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        train_ds = TextDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        valid_ds = TextDataset(valid_texts, valid_labels, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)
        history = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                out = self.model(**batch)
                out.loss.backward()
                optimizer.step()
                total_loss += out.loss.item()

            valid_probs, valid_labels_arr = self._eval_loader(valid_loader, device)
            preds = (valid_probs >= 0.5).astype(int)
            acc = float((preds == valid_labels_arr).mean())
            history.append({"epoch": epoch + 1, "train_loss": total_loss / max(1, len(train_loader)),
                             "valid_accuracy": acc})

        return history

    def _eval_loader(self, loader, device):
        import torch
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.numpy())
        return np.concatenate(all_probs), np.concatenate(all_labels)

    def predict_proba(self, texts, batch_size: int = None) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader

        if self.model is None:
            raise TransformerUnavailableError("Model not loaded/trained.")
        batch_size = batch_size or config.TRANSFORMER_BATCH_SIZE
        device = next(self.model.parameters()).device
        self.model.eval()

        probs = []
        texts = list(texts)
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = self.tokenizer(
                    chunk, truncation=True, padding=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(device)
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
        return np.concatenate(probs)

    def save(self, path: str = None):
        path = path or config.TRANSFORMER_MODEL_DIR
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str = None):
        _require_libs()
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = path or config.TRANSFORMER_MODEL_DIR
        if not os.path.isdir(path) or not os.listdir(path):
            raise TransformerUnavailableError(
                f"No trained transformer found at {path}. "
                "Run training/train_transformer.py first."
            )
        obj = cls.__new__(cls)
        obj.model_name = path
        obj.max_length = config.TRANSFORMER_MAX_LENGTH
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        return obj
