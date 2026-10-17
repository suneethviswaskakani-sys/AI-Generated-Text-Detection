"""
Centralized configuration for the AI Text Detection system.
Every tunable value lives here instead of being scattered across modules.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Dataset_AI_Human.csv")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
VALID_SIZE = 0.15
# Column used for the group-aware (leakage-controlled) split.
# NOTE: in this dataset, 'source' is almost perfectly collinear with the
# label (each generator source is ~100% AI or ~100% human), so grouping by
# 'source' is not a meaningful leakage test -- it degenerates into predicting
# the label from the group. 'prompt_name' (the essay topic) appears across
# both classes and is the column that actually tests generalization to
# unseen topics, so it is used for the group-aware split.
GROUP_COLUMN = "prompt_name"

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
MAX_WORD_FEATURES = 60000
MAX_CHAR_FEATURES = 50000
MIN_DOC_FREQ = 2
MAX_DOC_FREQ = 0.95
SUBLINEAR_TF = True

# FAST_MODE trades feature-space size / estimator count for speed on a single
# CPU (student-laptop-friendly, per project spec). Full-size settings above
# are used when FAST_MODE = False. Toggle via environment variable so
# training runs don't require editing this file.
import os as _os
FAST_MODE = _os.environ.get("FAST_MODE", "1") == "1"
if FAST_MODE:
    MAX_WORD_FEATURES = 15000
    MAX_CHAR_FEATURES = 8000
    MIN_DOC_FREQ = 3
    STRUCTURAL_N_ESTIMATORS_FAST = 120
    TRAIN_SAMPLE_SIZE = 16000  # stratified subsample of the cleaned dataset
    SVM_CALIBRATION_CV = 2
else:
    TRAIN_SAMPLE_SIZE = None
    SVM_CALIBRATION_CV = 3

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
WORD_MODEL_C = 3.0
CHAR_MODEL_C = 1.5
STRUCTURAL_N_ESTIMATORS = 120 if FAST_MODE else 250
STRUCTURAL_MAX_DEPTH = 4
STRUCTURAL_LEARNING_RATE = 0.08 if FAST_MODE else 0.05

# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------
ENSEMBLE_WEIGHTING = "auc"  # one of: "equal", "auc", "f1"
THRESHOLD_SEARCH_RANGE = (0.30, 0.70, 81)  # (start, stop, num_points)

# ---------------------------------------------------------------------------
# Reliability policy (by word count) -- see src/reliability.py
# ---------------------------------------------------------------------------
RELIABILITY_THRESHOLDS = {
    "insufficient": 50,   # below this: refuse a confident call
    "low": 100,           # below this: label result "low reliability"
    "moderate": 250,      # below this: label result "moderate reliability"
    # >= 250 words: "normal" reliability
}

# ---------------------------------------------------------------------------
# Optional advanced model (Model D): transformer classifier
# ---------------------------------------------------------------------------
# This project's sandbox has no internet access and no torch/transformers
# installed, so this model could not be trained or evaluated here (see
# src/transformer_model.py docstring and README "Phase 2" section for the
# honest account). The code is complete; run training/train_transformer.py
# yourself on a machine with internet access to produce real weights/metrics.
# Everything else in the app works unchanged with this set to False.
USE_TRANSFORMER = _os.environ.get("USE_TRANSFORMER", "0") == "1"
TRANSFORMER_MODEL_DIR = os.path.join(MODELS_DIR, "transformer_model")
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"  # small enough for local/CPU fine-tuning
TRANSFORMER_MAX_LENGTH = 256
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 2
TRANSFORMER_LR = 2e-5
# Fine-tuning a transformer on the full 44,868 rows is expensive; use a
# stratified subsample by default, same spirit as FAST_MODE above.
TRANSFORMER_TRAIN_SAMPLE_SIZE = 6000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_USER_TEXT = False  # never log raw submitted text by default (privacy)

AI_LABELS = {"AI", "AI-GENERATED", "AI GENERATED", "1", "TRUE"}
HUMAN_LABELS = {"HUMAN", "0", "FALSE"}
