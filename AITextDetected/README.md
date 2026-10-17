# 🛡️ VeriText AI Forensic Detector (Version 2.0)
### Industrial-Grade AI-Generated Text Detection, Sliding-Window Chunk Analysis, Mixed Authorship Identification & Explainable NLP

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-2.0%20Enterprise-red)
![Tests](https://img.shields.io/badge/Tests-46%20Passed%20(100%25)-brightgreen)
![Topic Leakage](https://img.shields.io/badge/Evaluation-Group--Aware%20Controlled-blueviolet)

---

## 🌟 Executive Overview

**VeriText 2.0** is an enterprise-grade NLP forensic system engineered to accurately detect, inspect, and explain AI-generated text in real-world documents (PDF, DOCX, TXT, CSV).

Unlike conventional college prototypes and opaque commercial tools that output brittle binary scores, VeriText operates as a **transparent multi-model forensic ensemble**. It integrates:
1. **Robust Document Ingestion**: Advanced PDF/DOCX structure recovery with frequency-based header/footer deduplication, line-wrap de-hyphenation, and bibliography section isolation.
2. **Sentence-Aligned Sliding-Window Chunking**: Splits multi-page documents into 280-word windows with 50-word overlaps, preventing feature explosion on long texts and pinpointing localized AI passages.
3. **Mixed Authorship Detection**: Identifies hybrid documents where human authors interspersed AI-generated paragraphs with human writing using inter-chunk score variance ($\text{Var} \ge 0.045$).
4. **4-Tier Uncertainty Classification**: Replaces brittle binary thresholds with calibrated decision boundaries (`AI-GENERATED`, `HUMAN-WRITTEN`, `MIXED CHARACTERISTICS`, `UNCERTAIN / LOW-CONFIDENCE`).
5. **Multi-Factor Reliability Assessment**: Scores prediction trustworthiness based on text length, extraction density, sub-model consensus, and boundary margins.
6. **Forensic Explainability (XAI)**: Provides exact lexical n-gram contributions, empirical stylometric corpus benchmarking, and leave-one-out feature occlusion attribution.

---

## 🏗️ Architectural Flow

```
                           +-------------------------------------+
                           |    Input Document: PDF / DOCX / TXT |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 1. Advanced Ingestion & Cleaning    |
                           |    - Header / Footer Deduplication  |
                           |    - Line-Wrap De-hyphenation       |
                           |    - Bibliography Section Isolation |
                           |    - Extraction Diagnostics & Grade |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 2. Sliding Window Chunking Engine   |
                           |    - Window: 280 words, 50 overlap  |
                           |    - Sentence Boundary Preservation |
                           +-------------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                        |                        |
                     v                        v                        v
            +------------------+     +------------------+     +------------------+
            | Model 1: Word LR |     | Model 2: Char SVM|     | Model 3: StructGB|
            | TF-IDF (1-2 gram)|     | TF-IDF (3-5 char)|     | 27 Stylometrics  |
            +------------------+     +------------------+     +------------------+
                     |                        |                        |
                     +------------------------+------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 3. Calibrated Multi-Model Ensemble  |
                           |    - Validation-AUC Optimal Weights |
                           |    - Model Agreement & Consensus    |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 4. Aggregation & Mixed Authorship   |
                           |    - Inter-Chunk Variance & Shifts  |
                           |    - Weighted Composite: Med+P75+Mean|
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 5. Explainability & Output Bundle   |
                           |    - AI Confidence Meter (0-100%)   |
                           |    - 4-Tier Uncertainty Label       |
                           |    - Multi-Factor Reliability Score |
                           |    - Occlusion Feature Attribution  |
                           |    - Lexical N-Gram Contributions   |
                           |    - Natural Language Summary       |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | 6. Dark Enterprise Streamlit 2.0 UI |
                           |    - 8 Specialized Forensic Views   |
                           |    - Batch Processing & CSV Reports |
                           +-------------------------------------+
```

---

## 📊 Scientific Evaluation & Leakage Control

VeriText was trained and evaluated on **44,868 real-world essays** from the Persuade Corpus and multi-generator benchmarks (ChatGPT, Mistral, LLaMA-2, Falcon, Claude).

To ensure complete scientific integrity, the model was evaluated under **two distinct protocols**:

| Evaluation Protocol | Description | Test Accuracy | Test F1-Score | Test ROC-AUC |
|---|---|:---:|:---:|:---:|
| **Random Stratified Split** | Standard baseline split | **99.28%** | **0.9907** | **0.9984** |
| **Group-Aware Split (`prompt_name`)** | **Strict Topic Leakage Control** (Evaluated on completely unseen essay prompts) | **97.40%** | **0.9484** | **0.9961** |

> [!NOTE]
> The modest $\sim 1.9\%$ drop between random split and group-aware split proves that the model learns genuine synthetic writing characteristics rather than memorizing prompt keywords.
> Brier calibration score on held-out test data is **0.0124**, verifying that the output probabilities reflect true empirical likelihood.

---

## 📁 Directory Structure

```
AITextDetected/
├── app.py                      # Modernized Streamlit 2.0 Dark Enterprise Application
├── predictor.py                # High-level forensic predictor with caching and chunking
├── config.py                   # Centralized configuration and hyperparameter settings
├── requirements.txt            # System dependencies
├── run_tests.py                # Standalone zero-dependency test runner
├── data/
│   ├── raw/
│   │   └── Dataset_AI_Human.csv# Curated corpus (44,868 rows)
│   └── reports/
│       ├── training_metrics.json # Full empirical evaluation metrics & error analysis
│       ├── dataset_report.json   # Dataset distributions and token statistics
│       └── experiments.csv       # Training experiment tracking log
├── models/
│   ├── word_vectorizer.joblib    # Fitted sublinear word TF-IDF vectorizer
│   ├── char_vectorizer.joblib    # Fitted sublinear character TF-IDF vectorizer
│   ├── word_model.joblib         # Calibrated Logistic Regression model
│   ├── char_model.joblib         # Calibrated LinearSVC (Sigmoid) model
│   ├── structural_scaler.joblib  # StandardScaler for 27 stylometric features
│   ├── structural_model.joblib   # Gradient Boosting stylometric classifier
│   ├── structural_feature_medians.joblib # Training medians for occlusion attribution
│   └── config.json               # Ensemble weights, thresholds, and feature names
├── reports/
│   ├── BUG_REPORT.md             # Forensic audit and root cause resolution report
│   ├── ARCHITECTURE_DOCUMENT.md  # System architecture and mathematical formulations
│   ├── MODEL_COMPARISON_AND_EVALUATION.md # In-depth empirical model comparisons
│   └── PRESENTATION_GUIDE.md     # University defense walkthrough and viva Q&A
├── src/
│   ├── __init__.py
│   ├── file_input.py             # Advanced PDF/DOCX/TXT parser with header/footer removal
│   ├── chunking.py               # Sliding-window engine and mixed authorship analyzer
│   ├── features.py               # 27 stylometric features and suspicious pattern detector
│   ├── ensemble.py               # Validation-AUC weighting and threshold optimization
│   ├── reliability.py            # Multi-factor reliability assessment engine
│   ├── explainability.py         # N-gram contributions, occlusion, and benchmarks
│   ├── data_pipeline.py          # Data ingestion, cleaning, and validation
│   ├── calibration_eval.py       # Brier score and reliability curve evaluation
│   ├── error_analysis.py         # False positive/negative profiling
│   ├── robustness.py             # Adversarial perturbation generators
│   └── utils.py                  # Logging and label normalization
└── tests/
    ├── __init__.py
    ├── test_v2_comprehensive.py  # 22 new forensic integration test cases
    ├── test_features.py          # Feature extraction unit tests
    ├── test_file_input.py        # File ingestion unit tests
    ├── test_ensemble.py          # Ensemble math unit tests
    └── test_robustness.py        # Perturbation and calibration unit tests
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- Standard terminal (PowerShell, Bash, or CMD).

### 2. Install Dependencies
```bash
cd AITextDetected
pip install -r requirements.txt
```

### 3. Run Automated Verification Tests
Verify all 46 automated unit and integration tests with zero external test dependencies:
```bash
python run_tests.py
```
*(Expected output: `Ran 46 tests in 0.6s ... OK [SUCCESS] ALL AUTOMATED TESTS PASSED!`)*

### 4. Launch the Streamlit Forensic Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 🖥️ Streamlit 2.0 Dashboard Features

The web interface is organized into **8 specialized forensic views**:
1. **📄 Document Analysis**: Upload PDF, DOCX, or TXT. Features real-time PDF extraction diagnostics (pages, words, removed headers, isolated references), the **AI Writing Confidence Meter (0–100%)**, 4-tier classification badge, and downloadable JSON forensic reports.
2. **🧩 Chunk-Level Inspector**: Interactive timeline showing chunk-by-chunk AI probability. Identifies mixed authorship and allows filtering by suspicious sections.
3. **🔍 Forensic Explainability**: Sub-model agreement bar chart, influential AI vs. Human lexical n-grams, stylometric corpus benchmarking, and leave-one-out feature occlusion attribution.
4. **✍️ Style & Pattern Detector**: Detects AI discourse markers (e.g. "moreover", "crucial role", "delve into"), evaluates sentence burstiness, and flags monotonous syntax.
5. **📂 Batch File Processing**: Upload multiple PDFs, DOCXs, TXTs, or a CSV file. Displays real-time progress and exports results to CSV.
6. **📊 Evaluation & Leakage Check**: Academic audit tab displaying real empirical metrics, confusion matrices, calibration curves, and topic leakage comparisons.
7. **⚙️ System & Model Health**: Matrix verification of all pipeline components, cache statistics, and memory controls.
8. **📜 Prediction History**: Chronological session history with timestamps, document names, scores, and CSV export.

---

## 🛠️ CLI Operations & Retraining

### Re-train Models (FAST_MODE)
To retrain and re-serialize the models natively on a laptop CPU:
```bash
python training/train.py
```

### Hyperparameter Tuning
To perform RandomizedSearchCV cross-validation tuning:
```bash
python training/tune.py
```

---

## 🎓 Academic Defense Highlights (For University Evaluation)

If presenting this project to a university guide or examination committee:
1. **Highlight the Root Cause Discovery**: Explain how naive unchunked extraction on a 5,000-word paper pushed length features $+30\sigma$ outside the training distribution, collapsing AI scores from $0.70$ down to $0.038$, and how sliding-window chunking solved this.
2. **Emphasize Topic Leakage Control**: Contrast the 99.28% random split accuracy against the 97.40% group-aware split accuracy, demonstrating that the system was tested on completely unseen essay topics.
3. **Showcase Explainability**: Demonstrate that every prediction is backed by mathematical proof: lexical token coefficients and leave-one-out occlusion attribution.
4. **Demonstrate Mixed Authorship**: Use the sidebar preset `🔀 Mixed Authorship Sample` to visually show how Section 1 is Human, Section 2 is AI, and Section 3 is Human.
