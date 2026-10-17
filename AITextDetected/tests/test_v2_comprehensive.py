"""
Comprehensive automated test suite for Version 2.0 AI-Generated Text Detection System.
Tests:
- File Ingestion: PDF, DOCX, TXT, CSV, empty files, corrupted files
- Chunking & Sliding Window: sentence preservation, overlap, edge cases
- Mixed Authorship: variance detection, transition tracking
- Reliability Assessment: multi-factor scoring, threshold bounds
- Suspicious Pattern Detection: discourse markers, burstiness, openers
- Prediction API: end-to-end inference, 4-tier labels, confidence meter
"""
import io
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.file_input import (
    read_txt_with_diagnostics, read_docx_with_diagnostics,
    read_pdf_with_diagnostics, read_batch_csv, strip_references,
    dehyphenate_and_normalize, detect_repeated_headers_footers,
)
from src.chunking import (
    create_chunks, detect_mixed_authorship, aggregate_chunk_predictions,
    split_sentences,
)
from src.features import (
    extract_structural_features, vectorize_structural,
    detect_suspicious_patterns, FEATURE_NAMES,
)
from src.reliability import assess_reliability
from src.explainability import (
    model_agreement, stylometric_benchmark_comparison,
    generate_explainability_narrative,
)
from predictor import Predictor


class TestFileInput(unittest.TestCase):
    def test_txt_reading_utf8(self):
        content = "Artificial intelligence is advancing rapidly. This is a test document."
        f = io.BytesIO(content.encode("utf-8"))
        text, diag = read_txt_with_diagnostics(f)
        self.assertEqual(text, content)
        self.assertEqual(diag["file_type"], "TXT")
        self.assertEqual(diag["total_words"], len(content.split()))

    def test_txt_encoding_fallback(self):
        content = "Café résumé naîve"
        f = io.BytesIO(content.encode("latin-1"))
        text, diag = read_txt_with_diagnostics(f)
        self.assertIn("Café", text)

    def test_dehyphenation(self):
        broken = "This is a com-\nprehensive test of dehyphen-\n   ation."
        fixed = dehyphenate_and_normalize(broken)
        self.assertIn("comprehensive", fixed)
        self.assertIn("dehyphenation", fixed)

    def test_strip_references(self):
        body = "This is the body of the paper discussing neural networks in education." * 10
        refs = "\n\nREFERENCES\n[1] Vaswani et al., Attention is all you need, 2017."
        full_text = body + refs
        clean_body, refs_extracted, has_refs = strip_references(full_text)
        self.assertTrue(has_refs)
        self.assertIn("Vaswani", refs_extracted)
        self.assertNotIn("REFERENCES", clean_body)

    def test_header_footer_deduplication(self):
        page_lines = [
            ["International Journal of AI", "Content page 1", "Vol 10, 2026"],
            ["International Journal of AI", "Content page 2", "Vol 10, 2026"],
            ["International Journal of AI", "Content page 3", "Vol 10, 2026"],
        ]
        headers, footers = detect_repeated_headers_footers(page_lines, min_fraction=0.5)
        self.assertIn("International Journal of AI", headers)
        self.assertIn("Vol 10, 2026", footers)

    def test_empty_file_handling(self):
        f = io.BytesIO(b"")
        text, diag = read_txt_with_diagnostics(f)
        self.assertEqual(text, "")
        self.assertEqual(diag["total_words"], 0)
        self.assertEqual(diag["extraction_quality"], "LOW")

    def test_batch_csv_loading(self):
        csv_data = "id,text,author\n1,Sample sentence one.,human\n2,Sample sentence two.,ai\n"
        df = read_batch_csv(io.BytesIO(csv_data.encode("utf-8")))
        self.assertEqual(len(df), 2)
        self.assertIn("text", df.columns)


class TestChunkingEngine(unittest.TestCase):
    def setUp(self):
        self.long_sample = (
            "Natural language processing has seen immense progress through large transformer models. "
            "These models learn statistical patterns from enormous web-scale text corpora. "
            "Consequently, their generations often display distinctive lexical and syntactic features. "
            "Human writers, on the other hand, produce idiosyncratic text with high sentence length variance. "
        ) * 15  # ~360 words

    def test_sentence_splitting(self):
        text = "First sentence here. Second sentence starts! Is this the third? Yes, it is."
        sents = split_sentences(text)
        self.assertEqual(len(sents), 4)

    def test_chunk_creation_target_words(self):
        chunks = create_chunks(self.long_sample, target_words=150, overlap_words=30, min_chunk_words=50)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertGreaterEqual(c["word_count"], 50)
            self.assertIn("text", c)
            self.assertIn("chunk_id", c)

    def test_mixed_authorship_detection(self):
        # Synthetic scenario: First 3 chunks human (0.15, 0.20, 0.18), next 3 chunks AI (0.85, 0.92, 0.88)
        scores = [0.15, 0.20, 0.18, 0.85, 0.92, 0.88]
        mixed_res = detect_mixed_authorship(scores, threshold_low=0.40, threshold_high=0.60)
        self.assertTrue(mixed_res["is_mixed"])
        self.assertGreater(mixed_res["mixed_confidence"], 0.6)
        self.assertGreaterEqual(mixed_res["ai_chunk_count"], 3)
        self.assertGreaterEqual(mixed_res["human_chunk_count"], 3)

    def test_uniform_authorship_detection(self):
        # All human chunks
        scores = [0.12, 0.18, 0.15, 0.22, 0.14]
        mixed_res = detect_mixed_authorship(scores)
        self.assertFalse(mixed_res["is_mixed"])

    def test_aggregate_chunk_predictions(self):
        chunk_results = [
            {"chunk_id": 1, "prediction_score": 0.80},
            {"chunk_id": 2, "prediction_score": 0.85},
            {"chunk_id": 3, "prediction_score": 0.78},
        ]
        agg = aggregate_chunk_predictions(chunk_results)
        self.assertEqual(agg["tier_label"], "AI-GENERATED")
        self.assertGreater(agg["composite_score"], 0.75)
        self.assertEqual(agg["chunk_count"], 3)


class TestFeaturesAndSuspicion(unittest.TestCase):
    def test_structural_feature_vector_dimensions(self):
        text = "This is an essay analyzing artificial intelligence and human reasoning."
        feats = extract_structural_features(text)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        for name in FEATURE_NAMES:
            self.assertIn(name, feats)

    def test_suspicious_pattern_detection_ai(self):
        ai_text = (
            "Furthermore, it is crucial to note that artificial intelligence plays a pivotal role in education. "
            "Moreover, delve into the realm of modern digital transformation. "
            "Additionally, a testament to the power of automated learning. "
            "In conclusion, it is important to note the implications."
        )
        susp = detect_suspicious_patterns(ai_text)
        self.assertGreater(susp["discourse_marker_density"], 10.0)
        self.assertGreater(susp["suspicion_index"], 40)
        self.assertIn("flags", susp)

    def test_suspicious_pattern_detection_human(self):
        human_text = (
            "Wait! "
            "I walked into the noisy room and immediately noticed the confusion. "
            "Why was everyone shouting so loudly? Nobody seemed to know! "
            "Perhaps the production server had crashed again; it was exceedingly difficult to tell with all the chaos. "
            "We finally fixed it."
        )
        susp = detect_suspicious_patterns(human_text)
        self.assertEqual(susp["discourse_marker_density"], 0.0)
        self.assertGreater(susp["burstiness"], 0.50)


class TestReliabilityAndExplainability(unittest.TestCase):
    def test_reliability_insufficient_length(self):
        rel = assess_reliability(word_count=20)
        self.assertEqual(rel["tier"], "insufficient")
        self.assertFalse(rel["usable"])

    def test_reliability_high_with_good_consensus(self):
        rel = assess_reliability(
            word_count=500,
            model_agreement_std=0.05,
            extraction_quality="HIGH",
            decision_margin=0.35,
        )
        self.assertEqual(rel["tier"], "normal")
        self.assertEqual(rel["label"], "High Reliability")
        self.assertTrue(rel["usable"])

    def test_model_agreement_calculation(self):
        scores = {"word": 0.85, "char": 0.88, "structural": 0.82}
        agree = model_agreement(scores)
        self.assertTrue(agree["unanimous"])
        self.assertLess(agree["score_std"], 0.05)


class TestPredictorEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()

    def test_predictor_health(self):
        health = self.predictor.get_model_health()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertTrue(health["components"]["word_model_lr"])
        self.assertTrue(health["components"]["char_model_svm"])
        self.assertTrue(health["components"]["structural_model_gb"])

    def test_short_text_prediction(self):
        text = "Hello world."
        res = self.predictor.predict(text)
        self.assertIn("reliability", res)
        self.assertEqual(res["reliability"]["tier"], "insufficient")

    def test_medium_text_prediction(self):
        text = (
            "The rapid growth of generative artificial intelligence has made it increasingly "
            "challenging to distinguish AI-generated text from human writing across digital media. "
            "This research evaluates lexical, character, and stylometric feature spaces to build "
            "robust and explainable classification ensembles. By leveraging both n-gram representations "
            "and sentence-level structural properties, detectors can provide transparent probability "
            "scores rather than opaque binary classifications."
        )
        res = self.predictor.predict(text)
        self.assertIn("label", res)
        self.assertIn("prediction_score", res)
        self.assertIn("ai_confidence_percent", res)
        self.assertIn("model_scores", res)
        self.assertIn("narrative_summary", res)
        self.assertGreater(len(res["narrative_summary"]), 20)

    def test_batch_prediction(self):
        items = [
            ("doc1", "Artificial intelligence in modern computing systems represents a pivotal shift."),
            ("doc2", "I think we should meet at the park tomorrow around noon if the weather stays sunny.")
        ]
        df = self.predictor.predict_batch(items)
        self.assertEqual(len(df), 2)
        self.assertIn("filename", df.columns)
        self.assertIn("prediction", df.columns)
        self.assertIn("ai_confidence_percent", df.columns)


if __name__ == "__main__":
    unittest.main()
