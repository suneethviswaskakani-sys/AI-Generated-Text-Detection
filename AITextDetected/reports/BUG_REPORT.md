# VeriText 2.0: Comprehensive Audit & Bug Report
## Root Cause Analysis and Engineering Resolutions

**Project**: Detection of AI-Generated Text Using a Hybrid Explainable Ensemble (Version 2.0)  
**Date**: October 2026  
**Auditor**: Senior Machine Learning & Systems Engineer  
**Status**: Resolved & Verified (100% Pass Rate across 46 Automated Unit & Integration Tests)

---

## 1. Executive Summary

A comprehensive forensic audit of the legacy AI text detection repository revealed critical architectural deficiencies, environment deserialization failures, and feature space distortions that severely degraded model performance on real-world inputs (particularly multi-page PDFs and academic papers).

Prior to this upgrade:
1. The application **crashed on startup** under modern Python / scikit-learn environments due to pickling incompatibility (`ModuleNotFoundError: No module named '_loss'`).
2. When evaluating real-world PDFs, the pipeline treated multi-page documents as **single monolithic strings**, skewing unbounded length features $+30$ standard deviations outside the training distribution and collapsing AI probability from $\sim 0.70$ down to $0.038$ (false negative).
3. Repeating headers, footers, page numbers, line-break hyphens, and bibliography sections heavily contaminated n-gram and stylometric calculations.
4. Classification was strictly binary ($0$ or $1$) with no uncertainty bands, confidence meters, or mixed-authorship detection.

All issues have been resolved in Version 2.0 through automated compatibility hooks, sentence-aligned sliding window chunking, multi-factor reliability assessment, and advanced PDF structure recovery.

---

## 2. Confirmed Bugs & Root Cause Analyses

### Bug #1: Critical Deserialization Crash on App Startup (`ModuleNotFoundError: No module named '_loss'`)
- **Severity**: Critical (P0 - Blocker)
- **Component**: `predictor.py`, `models/structural_model.joblib`
- **Root Cause**:
  `structural_model.joblib` was serialized with scikit-learn 1.8.0. Between scikit-learn 1.8 and 1.9, internal Cython loss implementations for `GradientBoostingClassifier` were refactored from `_loss.CyHalfBinomialLoss` to `sklearn._loss._loss.CyHalfBinomialLoss`. When loading with standard `joblib.load()`, Python attempted to resolve top-level module `_loss`, raising `ModuleNotFoundError` and terminating the Streamlit application before any user interaction could occur.
- **Resolution**:
  1. Injected a runtime compatibility bridge into `predictor.py` and model loaders:
     ```python
     try:
         import sklearn._loss._loss as cy_loss
         sys.modules['_loss'] = cy_loss
     except Exception:
         pass
     ```
  2. Executed a clean, native retraining cycle (`training/train.py`) that re-serialized all artifacts (`word_vectorizer.joblib`, `char_vectorizer.joblib`, `word_model.joblib`, `char_model.joblib`, `structural_scaler.joblib`, `structural_model.joblib`) natively under the active environment.

---

### Bug #2: Out-of-Distribution Feature Explosion on Multi-Page Documents
- **Severity**: Critical (P0 - Model Degradation)
- **Component**: `src/features.py`, `predictor.py`, `src/file_input.py`
- **Root Cause**:
  The models were trained on single student essays with an average length of $383$ words (training distribution: median $2,044$ characters, standard deviation $\sim 1,000$). When a $5,500$-word research paper PDF was parsed, `predictor.py` extracted structural features on the entire unchunked string. Unbounded raw counts (`char_count = 37,089`, `word_count = 5,920`, `sentence_count = 429`) were passed to `StandardScaler`, yielding $z$-scores exceeding $+34\sigma$. The gradient boosting decision trees had never encountered values in this range, causing `structural_gb` to output an AI probability of $0.0384$ and dragging the entire ensemble down to a false negative.
- **Resolution**:
  1. Engineered a **Sentence-Aligned Sliding Window Chunking Engine** (`src/chunking.py`).
  2. Documents exceeding $450$ words are split into window chunks of $280$ words ($50$-word overlap) along natural sentence boundaries.
  3. Every chunk is scored individually by the multi-model ensemble.
  4. Overall document probability is derived through robust composite aggregation (weighted combination of median, 75th percentile peak suspicion, and mean), maintaining feature stability within the model's intended operating envelope.

---

### Bug #3: Repeating PDF Headers & Footers Distorting Stylometric Repetition
- **Severity**: High (P1)
- **Component**: `src/file_input.py` (`read_pdf`)
- **Root Cause**:
  Legacy `read_pdf` joined raw `page.extract_text()` outputs with newline characters without filtering. Running headers (e.g. `[Author] et al.: Detection of AI-Generated Text Using a Hybrid Explainable Ensemble`) and running footers (e.g. `VOLUME 1, 2026 8`) appeared on every single page (9 times in the base paper). This artificially inflated repetitive bigrams (`bigram_repetition`), compressed `type_token_ratio`, and introduced repeated n-grams that contaminated lexical classification.
- **Resolution**:
  Engineered `detect_repeated_headers_footers` in `src/file_input.py`. The algorithm extracts the top 3 and bottom 3 lines of each page, identifies strings appearing across $\ge 40\%$ of pages, and automatically eliminates them along with standalone page numbers before linguistic normalization.

---

### Bug #4: Broken Line-Wrap Hyphenation Destructive to TF-IDF Vocabularies
- **Severity**: High (P1)
- **Component**: `src/file_input.py`
- **Root Cause**:
  Standard PDF column layout breaks words with trailing hyphens across line ends (e.g. `trans- \n former` or `ex- \n plainability`). The legacy parser left these as `trans-` and `former`, neither of which existed in the fitted 15,000-word TF-IDF vocabulary. Key technical terms were completely ignored by the linear classifier.
- **Resolution**:
  Added regex-based de-hyphenation in `dehyphenate_and_normalize`:
  ```python
  text = re.sub(r"(\b\w+)-\s*[\r\n]+\s*(\w+\b)", r"\1\2", text)
  ```
  All wrapped words are joined into legitimate dictionary tokens prior to vectorization.

---

### Bug #5: Bibliography / Citation Block Contamination
- **Severity**: High (P1)
- **Component**: `src/file_input.py`
- **Root Cause**:
  Academic papers terminate with dense reference sections (`REFERENCES [1] Vaswani et al....`). These sections consist of human author names, publication venues, years, and page ranges. Ingesting this into AI text detection heavily biased the lexical and character TF-IDF models towards human reference patterns, masking AI generation in the paper body.
- **Resolution**:
  Implemented `strip_references` in `src/file_input.py`. The system automatically isolates `REFERENCES`, `BIBLIOGRAPHY`, or `WORKS CITED` sections, calculates their word count for diagnostics, and excludes them from the core scoring pipeline.

---

### Bug #6: Binary-Only Overconfidence & Absence of Mixed Authorship Handling
- **Severity**: High (P1)
- **Component**: `predictor.py`, `app.py`
- **Root Cause**:
  Real-world student submissions and technical documents frequently exhibit hybrid authorship (e.g., human-authored introductions combined with AI-generated body sections). Legacy `predictor.py` used a single hard threshold (`score >= 0.44`), collapsing nuanced probability distributions into a single binary call ("AI-GENERATED" or "HUMAN-WRITTEN").
- **Resolution**:
  Implemented a **4-Tier Uncertainty Classification System**:
  - `AI-GENERATED` (Score $\ge 0.58$ with strong consensus)
  - `HUMAN-WRITTEN` (Score $\le 0.42$ with strong consensus)
  - `MIXED CHARACTERISTICS` (Triggered when inter-chunk score variance $\ge 0.045$ with both AI and Human sections present)
  - `UNCERTAIN / LOW-CONFIDENCE` (Borderline probability $[0.42, 0.58]$ or insufficient sample density)
  Accompanied by a continuous **AI Writing Confidence Meter (0–100%)**.

---

### Bug #7: Test Suite Incompatibility (`pytest` Not Installed)
- **Severity**: Medium (P2)
- **Component**: `tests/`
- **Root Cause**:
  Legacy tests relied on bare functions (`def test_...(): assert ...`) designed for `pytest`. In standard environments where `pytest` is not installed, executing `python -m unittest` discovered $0$ tests.
- **Resolution**:
  1. Converted all test suites to inherit from Python's standard `unittest.TestCase`.
  2. Authored `tests/test_v2_comprehensive.py` containing 22 new forensic integration tests.
  3. Created `run_tests.py`, a zero-dependency standalone runner executing all 46 test cases.

---

## 3. Possible Bugs & Edge Cases Investigated

| Issue | Investigation Finding | Action Taken |
|---|---|---|
| **Scanned / Image PDFs** | If a PDF contains scanned bitmap pages, `pypdf` extracts empty strings without alerting the user. | Added `blank_pages` counter and extraction quality grading (`POOR / OCR NEEDED` warning). |
| **Short Text Fragility** | Inputs $<50$ words lack sufficient n-gram frequency density. | Upgraded `src/reliability.py` to flag insufficient text and suppress confident verdicts. |
| **Topic Leakage** | Random train/test split allows near-duplicate essay prompts across splits, inflating apparent accuracy. | Evaluated with Group-Aware Split on `prompt_name`; both random and group-aware metrics reported transparently. |

---

## 4. Verification & Status Summary

- **Total Automated Tests**: 46
- **Test Results**: 46 Passed, 0 Failures, 0 Errors (100% Success Rate)
- **PDF Extraction Verification**: Tested on IEEE format `AI_Text_Detection_Base_Paper.pdf`:
  - 9 pages extracted
  - 9 repeated headers removed
  - 629 reference words isolated
  - 18 sentence-aligned chunks generated
  - Chunked score: `0.6335` (`AI-GENERATED`), resolving the legacy `0.220` false negative.
