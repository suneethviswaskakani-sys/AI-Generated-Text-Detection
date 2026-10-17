"""
Inference-time predictor for Version 2.0 AI-Generated Text Detection.
Loads persisted models with automated backward-compatibility deserialization,
runs single or chunk-level multi-model ensembles, assesses multi-factor reliability,
evaluates mixed authorship, and outputs rich forensic explainability.
"""
import hashlib
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

# Deserialization compatibility hook for scikit-learn 1.8 <-> 1.9
try:
    import sklearn._loss._loss as cy_loss
    sys.modules['_loss'] = cy_loss
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.features import (
    vectorize_structural, extract_structural_features, FEATURE_NAMES,
    detect_suspicious_patterns,
)
from src.reliability import assess_reliability
from src.chunking import create_chunks, aggregate_chunk_predictions
from src.explainability import (
    top_word_contributors, structural_feature_report, model_agreement,
    occlusion_attribution, stylometric_benchmark_comparison,
    generate_explainability_narrative,
)
from src.utils import get_logger

logger = get_logger("predictor")


class ModelLoadError(RuntimeError):
    pass


class Predictor:
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or config.MODELS_DIR
        self.transformer = None
        self.advanced_weights = None
        self.advanced_threshold = None
        self.feature_medians = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 128
        self._load()

    def _path(self, name: str) -> str:
        return os.path.join(self.models_dir, name)

    def _load(self):
        required = [
            "word_vectorizer.joblib", "char_vectorizer.joblib", "word_model.joblib",
            "char_model.joblib", "structural_scaler.joblib", "structural_model.joblib",
            "config.json",
        ]
        missing = [f for f in required if not os.path.exists(self._path(f))]
        if missing:
            raise ModelLoadError(
                f"Missing model artifact(s): {missing}. Run `python training/train.py` first."
            )
        try:
            self.word_vec = joblib.load(self._path("word_vectorizer.joblib"))
            self.char_vec = joblib.load(self._path("char_vectorizer.joblib"))
            self.word_model = joblib.load(self._path("word_model.joblib"))
            self.char_model = joblib.load(self._path("char_model.joblib"))
            self.scaler = joblib.load(self._path("structural_scaler.joblib"))
            self.structural_model = joblib.load(self._path("structural_model.joblib"))
            with open(self._path("config.json")) as f:
                mc = json.load(f)
            self.weights = mc["weights"]
            self.threshold = mc["threshold"]
        except Exception as e:
            raise ModelLoadError(f"Failed to load model artifacts: {e}") from e
        logger.info("Base models loaded successfully.")

        medians_path = self._path("structural_feature_medians.joblib")
        if os.path.exists(medians_path):
            self.feature_medians = joblib.load(medians_path)

        if config.USE_TRANSFORMER:
            try:
                from src.transformer_model import TransformerTextClassifier
                self.transformer = TransformerTextClassifier.load()
                self.advanced_weights = mc.get("advanced_ensemble_weights")
                self.advanced_threshold = mc.get("advanced_ensemble_threshold")
                if not self.advanced_weights:
                    raise ModelLoadError(
                        "Transformer weights found but no advanced_ensemble_weights in "
                        "models/config.json -- run training/train_transformer.py."
                    )
                logger.info("Transformer model loaded; using advanced 4-model ensemble.")
            except Exception as e:
                logger.info(f"Transformer not available ({e}); using 3-model baseline ensemble.")
                self.transformer = None

    def get_model_health(self) -> Dict[str, Any]:
        """Verify model artifacts and return operational health status."""
        health = {
            "status": "HEALTHY",
            "models_dir": self.models_dir,
            "components": {
                "word_vectorizer": self.word_vec is not None,
                "char_vectorizer": self.char_vec is not None,
                "word_model_lr": self.word_model is not None,
                "char_model_svm": self.char_model is not None,
                "structural_scaler": self.scaler is not None,
                "structural_model_gb": self.structural_model is not None,
                "feature_medians": self.feature_medians is not None,
                "transformer_active": self.transformer is not None,
            },
            "ensemble_weights": self.weights,
            "decision_threshold": self.threshold,
            "active_ensemble_type": "advanced_4_model" if self.transformer else "baseline_3_model",
            "cache_entries": len(self._cache),
        }
        return health

    def _score_single_segment(self, text: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float], float, float, str]:
        """Compute raw model scores for a single text segment (chunk or short text)."""
        Xw = self.word_vec.transform([text])
        Xc = self.char_vec.transform([text])
        sw = vectorize_structural([text])
        A = self.scaler.transform(sw)

        pw = float(self.word_model.predict_proba(Xw)[0, 1])
        pc = float(self.char_model.predict_proba(Xc)[0, 1])
        ps = float(self.structural_model.predict_proba(A)[0, 1])

        model_scores = {
            "word_tfidf_lr": pw,
            "char_tfidf_svm": pc,
            "structural_gb": ps,
        }
        agreement_input = {"word": pw, "char": pc, "structural": ps}

        if self.transformer is not None:
            pt = float(self.transformer.predict_proba([text])[0])
            model_scores["transformer"] = pt
            agreement_input["transformer"] = pt
            w = self.advanced_weights
            final = (w["word_tfidf_lr"] * pw + w["char_tfidf_svm"] * pc
                     + w["structural_gb"] * ps + w["transformer"] * pt)
            threshold = self.advanced_threshold
            ensemble_type = "advanced_4_model"
        else:
            w = self.weights
            final = w["word_tfidf_lr"] * pw + w["char_tfidf_svm"] * pc + w["structural_gb"] * ps
            threshold = self.threshold
            ensemble_type = "baseline_3_model"

        return sw, model_scores, agreement_input, w, final, threshold, ensemble_type

    def predict(
        self,
        text: str,
        diagnostics: Optional[Dict[str, Any]] = None,
        enable_chunking: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Main forensic prediction method.
        - Employs sliding-window chunking for long documents (>= 450 words) to avoid
          out-of-distribution feature distortions and identify localized AI text.
        - Provides multi-tier classification: AI-Generated, Human-Written, Mixed Characteristics, Uncertain.
        - Calculates AI Writing Confidence Meter (0-100%).
        - Assesses multi-factor reliability and returns exhaustive explainability.
        """
        text = (text or "").strip()
        word_count = len(text.split())

        # Check cache
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest() if use_cache else None
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key].copy()
            if diagnostics:
                cached["diagnostics"] = diagnostics
            return cached

        # Extraction quality from diagnostics if present
        ext_quality = diagnostics.get("extraction_quality", "HIGH") if diagnostics else "HIGH"

        if not text:
            rel = assess_reliability(0, extraction_quality=ext_quality)
            return {
                "error": "Empty text provided.",
                "label": "INSUFFICIENT TEXT",
                "prediction_score": 0.5,
                "ai_confidence_percent": 50.0,
                "reliability": rel,
                "diagnostics": diagnostics,
            }

        start_time = time.time()

        # Decide whether to chunk
        should_chunk = enable_chunking and word_count >= 450
        chunks_info = []
        chunk_results = []

        if should_chunk:
            raw_chunks = create_chunks(text, target_words=280, overlap_words=50, min_chunk_words=70)
            for c in raw_chunks:
                _, m_scores, c_agree_in, _, c_final, c_thresh, _ = self._score_single_segment(c["text"])
                c_agree = model_agreement(c_agree_in)
                c_label = "AI-GENERATED" if c_final >= 0.55 else ("HUMAN-WRITTEN" if c_final <= 0.45 else "UNCERTAIN")
                chunk_entry = {
                    "chunk_id": c["chunk_id"],
                    "word_count": c["word_count"],
                    "start_word": c["start_word"],
                    "end_word": c["end_word"],
                    "text_preview": c["text"][:120] + "...",
                    "full_text": c["text"],
                    "prediction_score": round(c_final, 4),
                    "label": c_label,
                    "model_scores": {k: round(v, 4) for k, v in m_scores.items()},
                    "agreement_std": round(c_agree["score_std"], 3),
                    "is_suspicious": c_final >= 0.55,
                }
                chunk_results.append(chunk_entry)

            chunk_summary = aggregate_chunk_predictions(chunk_results, decision_threshold=self.threshold)
            final = chunk_summary["composite_score"]
            threshold = self.threshold
            ensemble_type = "chunked_sliding_window"
            is_mixed = chunk_summary["mixed_analysis"]["is_mixed"]
            label = chunk_summary["tier_label"]
            
            # Representative segment for explainability (top suspicious chunk or first chunk)
            rep_chunk = chunk_summary["suspicious_chunks"][0]["full_text"] if chunk_summary["suspicious_chunks"] else raw_chunks[0]["text"]
            sw, model_scores, agreement_input, w, _, _, _ = self._score_single_segment(rep_chunk)
        else:
            sw, model_scores, agreement_input, w, final, threshold, ensemble_type = self._score_single_segment(text)
            chunk_summary = {
                "composite_score": round(final, 4),
                "mean_score": round(final, 4),
                "median_score": round(final, 4),
                "max_score": round(final, 4),
                "min_score": round(final, 4),
                "iqr": 0.0,
                "tier_label": "AI-GENERATED" if final >= 0.58 else ("HUMAN-WRITTEN" if final <= 0.42 else "UNCERTAIN / LOW-CONFIDENCE"),
                "calibrated_confidence": round(abs(final - 0.5) * 2.0, 3),
                "confidence_percentage": round(final * 100.0, 1),
                "chunk_count": 1,
                "mixed_analysis": {"is_mixed": False, "assessment": "Single segment; chunking not required."},
                "suspicious_chunks": [],
            }
            label = chunk_summary["tier_label"]
            is_mixed = False

        agreement = model_agreement(agreement_input)
        decision_margin = abs(final - threshold)
        reliability = assess_reliability(
            word_count=word_count,
            model_agreement_std=agreement["score_std"],
            extraction_quality=ext_quality,
            decision_margin=decision_margin,
        )

        if not reliability["usable"]:
            label = "UNCERTAIN / LOW-CONFIDENCE"

        # Forensic Explainability
        explain_words = top_word_contributors(self.word_vec, self.word_model, text, top_k=10)
        explain_struct = structural_feature_report(
            self.structural_model, FEATURE_NAMES, sw.iloc[0].values, top_k=8
        )
        explain_occlusion = []
        if self.feature_medians is not None:
            explain_occlusion = occlusion_attribution(
                self.structural_model, self.scaler, FEATURE_NAMES,
                sw.iloc[0].values, self.feature_medians, top_k=8
            )

        suspicious_patterns = detect_suspicious_patterns(text)
        benchmarks = stylometric_benchmark_comparison(sw.iloc[0].to_dict())
        narrative = generate_explainability_narrative(
            prediction_score=final,
            label=label,
            top_ai_words=explain_words.get("ai_leaning", []),
            top_human_words=explain_words.get("human_leaning", []),
            top_structural_occlusion=explain_occlusion,
            agreement=agreement,
            is_mixed=is_mixed,
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        result = {
            "label": label,
            "prediction_score": round(final, 4),
            "ai_confidence_percent": round(final * 100.0, 1),
            "decision_strength": round(abs(final - 0.5) * 2.0, 3),
            "threshold_used": threshold,
            "ensemble_type": ensemble_type,
            "word_count": word_count,
            "char_count": len(text),
            "reliability": reliability,
            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "model_weights": w,
            "model_agreement": agreement,
            "is_mixed_authorship": is_mixed,
            "chunks": chunk_results,
            "chunk_summary": chunk_summary,
            "explain_words": explain_words,
            "explain_structural_features": explain_struct,
            "explain_occlusion": explain_occlusion,
            "suspicious_patterns": suspicious_patterns,
            "stylometric_benchmarks": benchmarks,
            "narrative_summary": narrative,
            "diagnostics": diagnostics or {},
            "inference_time_ms": round(elapsed_ms, 1),
        }

        # Cache result
        if use_cache and cache_key:
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = result

        return result

    def predict_batch(self, items: List[Union[str, Tuple[str, str]]]) -> pd.DataFrame:
        """
        Run predict() across multiple texts or (filename, text) pairs.
        Returns a rich DataFrame suitable for CSV export and reporting.
        """
        rows = []
        for item in items:
            if isinstance(item, tuple):
                filename, text = item
            else:
                filename, text = "input_text", item

            res = self.predict(text)
            if "error" in res:
                rows.append({
                    "filename": filename,
                    "prediction": "ERROR",
                    "ai_confidence_percent": None,
                    "score": None,
                    "reliability": None,
                    "mixed_authorship": None,
                    "word_count": 0,
                    "chunk_count": 0,
                    "word_lr_score": None,
                    "char_svm_score": None,
                    "structural_gb_score": None,
                    "transformer_score": None,
                    "inference_time_ms": None,
                })
                continue

            ms = res["model_scores"]
            rows.append({
                "filename": filename,
                "prediction": res["label"],
                "ai_confidence_percent": res["ai_confidence_percent"],
                "score": res["prediction_score"],
                "reliability": res["reliability"]["label"],
                "mixed_authorship": "YES" if res["is_mixed_authorship"] else "NO",
                "word_count": res["word_count"],
                "chunk_count": len(res["chunks"]) if res["chunks"] else 1,
                "word_lr_score": ms.get("word_tfidf_lr"),
                "char_svm_score": ms.get("char_tfidf_svm"),
                "structural_gb_score": ms.get("structural_gb"),
                "transformer_score": ms.get("transformer"),
                "inference_time_ms": res["inference_time_ms"],
            })

        return pd.DataFrame(rows)
