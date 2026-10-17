"""
Streamlit Application: AI-Generated Text Detection & Explainability System (Version 2.0).
Enterprise-grade forensic dashboard for analyzing documents (PDF, DOCX, TXT, CSV),
detecting mixed authorship, inspecting chunk-level signals, and providing
transparent multi-model explainability.
"""
import datetime
import io
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="VeriText AI Forensic Detector 2.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Insert root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from predictor import Predictor, ModelLoadError
from src.file_input import (
    load_uploaded_file, read_batch_csv, read_pdf_with_diagnostics,
    read_docx_with_diagnostics, read_txt_with_diagnostics,
)
from src.features import detect_suspicious_patterns

# ---------------------------------------------------------------------------
# Custom CSS for Dark Enterprise Theme & Glassmorphism
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global styling */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    
    /* Header branding */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #161f30;
        border: 1px solid #2d3b55;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Classification Badges */
    .badge-ai {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-human {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-mixed {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-uncertain {
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid #64748b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }

    /* Diagnostics Card */
    .diag-box {
        background: #111a28;
        border: 1px solid #253347;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .diag-title {
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }

    /* Chunk cards */
    .chunk-card {
        background: #162032;
        border: 1px solid #28374d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .chunk-card-suspicious {
        background: #23161c;
        border: 1px solid #842029;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Confidence Meter bar */
    .meter-container {
        width: 100%;
        background-color: #212c3d;
        border-radius: 12px;
        height: 24px;
        margin: 0.8rem 0;
        overflow: hidden;
        border: 1px solid #33445c;
    }
    .meter-bar {
        height: 100%;
        text-align: right;
        padding-right: 8px;
        line-height: 24px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        transition: width 0.5s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Management & Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor_instance() -> Predictor:
    return Predictor()


def init_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "last_diag" not in st.session_state:
        st.session_state["last_diag"] = None
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = None


init_session_state()

# ---------------------------------------------------------------------------
# Load Predictor
# ---------------------------------------------------------------------------
try:
    predictor = get_predictor_instance()
    health_status = predictor.get_model_health()
except ModelLoadError as e:
    st.error(f"Critical Model Loading Error: {e}")
    st.info("Please run `python training/train.py` to compile native model artifacts.")
    st.stop()


# ---------------------------------------------------------------------------
# Sample Pre-loads
# ---------------------------------------------------------------------------
SAMPLE_HUMAN = (
    "When I reflect on the years spent learning computer science, the most striking lesson "
    "was not the mathematical purity of algorithms, but the sheer chaos of debugging at two in the morning. "
    "I remember staring at an unresponsive terminal in the campus library, convinced that the compiler itself "
    "had developed a personal vendetta against my code. It turned out to be an extra comma in a JSON configuration file. "
    "Moments like that teach humility faster than any textbook ever could. Technology, at its core, is a deeply human "
    "and messy endeavor, full of quirks, late-night epiphanies, and hard-earned lessons."
)

SAMPLE_AI = (
    "Furthermore, the integration of generative artificial intelligence into academic ecosystems plays a crucial role "
    "in fostering multidisciplinary innovation. Moreover, it is important to note that neural language models utilize "
    "deep transformer architectures to seamlessly analyze contextual token distributions across extensive corpora. "
    "Consequently, educators must acknowledge the paramount significance of digital literacy in modern pedagogical frameworks. "
    "In conclusion, by exploring the synergistic potential of algorithmic reasoning, higher education can effectively bridge "
    "the gap between traditional instructional paradigms and emerging technological frontiers."
)

SAMPLE_MIXED = (
    "I began this project with a simple question: can machine learning models genuinely understand student essays, "
    "or are they merely mimicking superficial statistical patterns? During our initial experiments in the lab, "
    "we noticed that standard linear classifiers frequently stumbled on colloquial writing and idiomatic expressions. "
    "\n\n"
    "Furthermore, it is crucial to recognize that advanced sequence classifiers harness the power of contextual embeddings "
    "to mitigate syntactic ambiguity. Moreover, the integration of attention mechanisms plays a pivotal role in capturing "
    "long-range sequential dependencies across dense multidimensional representations. Consequently, modern paradigms "
    "provide a testament to the transformative capacity of automated cognitive systems in contemporary discourse. "
    "\n\n"
    "Yet when we evaluated these complex networks against noisy student submissions, we realized that fancy embeddings "
    "cannot replace careful error analysis and domain grounding. Ultimately, we must balance algorithmic sophistication "
    "with practical common sense."
)


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### **VeriText Forensic Engine**")
    st.caption("Version 2.0 • Hybrid NLP Ensemble")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "📄 Document Analysis",
            "🧩 Chunk-Level Inspector",
            "🔍 Forensic Explainability",
            "✍️ Style & Pattern Detector",
            "📂 Batch File Processing",
            "📊 Evaluation & Leakage Check",
            "⚙️ System & Model Health",
            "📜 Prediction History",
        ],
    )

    st.markdown("---")
    st.markdown("#### **Preset Demonstrations**")
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if st.button("🧑 Human Text", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_HUMAN
            st.session_state["last_diag"] = None
    with c_s2:
        if st.button("🤖 AI Sample", use_container_width=True):
            st.session_state["input_text"] = SAMPLE_AI
            st.session_state["last_diag"] = None

    if st.button("🔀 Mixed Authorship Sample", use_container_width=True):
        st.session_state["input_text"] = SAMPLE_MIXED
        st.session_state["last_diag"] = None

    st.markdown("---")
    st.markdown("#### **Engine Health**")
    st.markdown(
        f"<span style='color: #4ade80;'>●</span> **Ensemble Active** ({health_status['active_ensemble_type']})<br>"
        f"<small style='color: #94a3b8;'>Models: Word LR, Char SVM, Structural GB</small>",
        unsafe_allow_html=True,
    )
    st.caption(f"Session Queries: {len(st.session_state['history'])}")


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1 class="main-title">🛡️ VeriText AI Forensic Detector 2.0</h1>
        <div class="sub-title">
            Industrial-Grade AI-Generated Text Detection, Sliding-Window Chunk Analysis, 
            Mixed Authorship Identification & Multi-Model Explainability
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# VIEW 1: DOCUMENT ANALYSIS
# ===========================================================================
if nav_selection == "📄 Document Analysis":
    st.markdown("### Document Ingestion & Forensic Analysis")
    
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_doc = st.file_uploader(
            "Upload Document (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            help="Full document extraction with automatic header/footer deduplication, dehyphenation, and citation isolation.",
        )
        if uploaded_doc is not None:
            try:
                extracted_txt, doc_diag = load_uploaded_file(uploaded_doc)
                st.session_state["input_text"] = extracted_txt
                st.session_state["last_diag"] = doc_diag
                st.success(f"Successfully loaded '{uploaded_doc.name}' ({doc_diag['total_words']} words).")
            except Exception as ex:
                st.error(f"Error parsing uploaded file: {ex}")

    with col_info:
        st.markdown(
            """
            <div style="background: #151d2a; padding: 1rem; border-radius: 8px; border: 1px solid #28374d; font-size: 0.85rem;">
                <b>Supported Formats & Processing:</b>
                <ul style="margin: 0.5rem 0 0 1.2rem; padding: 0; color: #94a3b8;">
                    <li><b>PDF</b>: Header/footer removal, de-hyphenation</li>
                    <li><b>DOCX</b>: Paragraph and table parsing</li>
                    <li><b>TXT</b>: Multi-encoding recovery (UTF-8, Latin-1)</li>
                    <li><b>Chunking</b>: Automatic for documents &ge; 450 words</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document text area
    user_text = st.text_area(
        "Document Content to Analyze",
        value=st.session_state.get("input_text", ""),
        height=220,
        placeholder="Paste document text here or upload a file above...",
    )

    # Document Extraction Diagnostics Box
    diag = st.session_state.get("last_diag")
    if diag and diag.get("file_type") == "PDF":
        q_color = "#4ade80" if diag["extraction_quality"] == "HIGH" else "#f59e0b"
        st.markdown(
            f"""
            <div class="diag-box">
                <div class="diag-title">📄 PDF Extraction Diagnostics & Cleaning Summary</div>
                <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
                    <div><b>Pages Processed:</b> {diag['page_count']}</div>
                    <div><b>Words Extracted:</b> {diag['total_words']:,}</div>
                    <div><b>Quality Grade:</b> <span style="color: {q_color}; font-weight: bold;">{diag['extraction_quality']}</span></div>
                    <div><b>Headers Removed:</b> {diag['removed_headers_count']} instances</div>
                    <div><b>References Isolated:</b> {'Yes (' + str(diag['references_word_count']) + ' words)' if diag['references_isolated'] else 'No'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if diag.get("warnings"):
            for w in diag["warnings"]:
                st.caption(f"⚠️ {w}")

    col_btn, col_opt = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("🚀 Analyze Document", type="primary", use_container_width=True)
    with col_opt:
        enable_chunk_opt = st.checkbox("Enable Sentence-Aligned Sliding Window Chunking", value=True)

    if run_analysis:
        if not user_text.strip():
            st.warning("Please provide or upload document text before running analysis.")
        else:
            with st.spinner("Executing multi-model forensic analysis & chunk evaluation..."):
                res = predictor.predict(
                    user_text,
                    diagnostics=st.session_state.get("last_diag"),
                    enable_chunking=enable_chunk_opt,
                )
                st.session_state["last_result"] = res
                
                # Append to prediction history
                filename = uploaded_doc.name if uploaded_doc else "Direct Text Input"
                st.session_state["history"].append({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": filename,
                    "prediction": res["label"],
                    "ai_confidence_percent": res["ai_confidence_percent"],
                    "word_count": res["word_count"],
                    "reliability": res["reliability"]["label"],
                    "chunks": len(res.get("chunks", [])),
                    "mixed": "YES" if res.get("is_mixed_authorship") else "NO",
                })

    # Render Analysis Results
    res = st.session_state.get("last_result")
    if res and "error" not in res:
        st.markdown("---")
        st.markdown("### Forensic Assessment Summary")

        # Top Banner: Badge, Score, Reliability
        b_col1, b_col2, b_col3, b_col4 = st.columns([1.5, 1.2, 1.2, 1.2])
        
        # Badge rendering
        lbl = res["label"]
        if "AI-GENERATED" in lbl:
            badge_html = f"<div class='badge-ai'>🤖 {lbl}</div>"
        elif "HUMAN-WRITTEN" in lbl:
            badge_html = f"<div class='badge-human'>🧑 {lbl}</div>"
        elif "MIXED" in lbl:
            badge_html = f"<div class='badge-mixed'>🔀 {lbl}</div>"
        else:
            badge_html = f"<div class='badge-uncertain'>⚖️ {lbl}</div>"

        with b_col1:
            st.markdown(badge_html, unsafe_allow_html=True)
            st.caption(f"Inference Time: {res['inference_time_ms']} ms • Ensemble: {res['ensemble_type']}")

        with b_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">AI Writing Likelihood</div>
                    <div class="metric-val" style="color: {'#f87171' if res['ai_confidence_percent']>=55 else ('#4ade80' if res['ai_confidence_percent']<=45 else '#fbbf24')};">
                        {res['ai_confidence_percent']:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b_col3:
            rel = res["reliability"]
            rel_color = "#4ade80" if rel["tier"] == "normal" else ("#fbbf24" if rel["tier"] == "moderate" else "#f87171")
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Reliability Tier</div>
                    <div class="metric-val" style="color: {rel_color}; font-size: 1.4rem; padding-top: 0.4rem;">
                        {rel['label']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with b_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Document Length</div>
                    <div class="metric-val" style="color: #60a5fa; font-size: 1.4rem; padding-top: 0.4rem;">
                        {res['word_count']:,} <span style="font-size: 0.9rem; font-weight: normal; color: #94a3b8;">words</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # AI Confidence Visual Meter
        meter_pct = max(0, min(100, res["ai_confidence_percent"]))
        meter_color = (
            "linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%)"
            if meter_pct >= 50
            else "linear-gradient(90deg, #22c55e 0%, #3b82f6 100%)"
        )
        st.markdown(
            f"""
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8;">
                    <span>0% (Human-Authored)</span>
                    <span><b>AI Confidence Meter: {meter_pct:.1f}%</b></span>
                    <span>100% (Synthetic AI)</span>
                </div>
                <div class="meter-container">
                    <div class="meter-bar" style="width: {meter_pct}%; background: {meter_color};">
                        {meter_pct:.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Narrative Summary Callout
        st.markdown(
            f"""
            <div style="background: #151e2e; border-left: 4px solid #60a5fa; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1.2rem 0;">
                <b style="color: #60a5fa;">Analytical Forensic Summary:</b><br>
                <span style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">{res['narrative_summary']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Highlights Columns
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.markdown("##### 📌 Model Agreement Consensus")
            ag = res["model_agreement"]
            st.write(f"- Consensus Agreement: **{ag['agreement_fraction']*100:.0f}%**")
            st.write(f"- Cross-Model Spread (Std): **{ag['score_std']:.3f}**")
            st.write(f"- Unanimous Verdict: **{'Yes' if ag['unanimous'] else 'Divergent'}**")

        with h_col2:
            st.markdown("##### 🔀 Authorship Structure")
            if res.get("is_mixed_authorship"):
                st.warning("⚠️ Mixed authorship detected across sections.")
            else:
                st.success("✅ Stylistic consistency uniform throughout.")
            if res.get("chunks"):
                st.write(f"- Total Analyzed Chunks: **{len(res['chunks'])}**")
                st.write(f"- Suspicious Chunks: **{len([c for c in res['chunks'] if c['is_suspicious']])}**")

        with h_col3:
            st.markdown("##### 🔍 Stylometric Indicators")
            sp = res["suspicious_patterns"]
            st.write(f"- Sentence Burstiness: **{sp['burstiness']:.2f}** ({sp['burstiness_assessment']})")
            st.write(f"- AI Marker Density: **{sp['discourse_marker_density']:.1f}** / 1k words")
            st.write(f"- Synthetic Suspicion Index: **{sp['suspicion_index']} / 100**")

        # Download Full Report Button
        st.markdown("---")
        report_json = json.dumps(res, indent=2, default=str)
        st.download_button(
            label="📥 Download Comprehensive Forensic Report (JSON)",
            data=report_json,
            file_name=f"veritext_forensic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# ===========================================================================
# VIEW 2: CHUNK-LEVEL INSPECTOR
# ===========================================================================
elif nav_selection == "🧩 Chunk-Level Inspector":
    st.markdown("### 🧩 Sentence-Aligned Chunk Inspector & Mixed Authorship")
    res = st.session_state.get("last_result")

    if not res:
        st.info("Please run an analysis in the 'Document Analysis' tab first.")
    elif not res.get("chunks"):
        st.warning("Chunking was not triggered (text is short, < 450 words). Text was evaluated as a single coherent unit.")
    else:
        chunks = res["chunks"]
        chunk_sum = res["chunk_summary"]
        mixed = chunk_sum["mixed_analysis"]

        # Mixed Authorship Callout Banner
        if mixed["is_mixed"]:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1.2rem;">
                    <h4 style="color: #fbbf24; margin: 0 0 0.4rem 0;">⚠️ Mixed Authorship Identified</h4>
                    <p style="margin: 0; color: #fde68a;">{mixed['assessment']}</p>
                    <small style="color: #fbbf24;">Mixed Confidence: {mixed['mixed_confidence']*100:.0f}% • Transitions: {mixed['transitions']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success(f"✅ Stylistic Uniformity Confirmed: {mixed['assessment']}")

        # Chunk Timeline Chart
        st.markdown("#### **Section Progression Timeline**")
        timeline_df = pd.DataFrame([
            {
                "Chunk": f"Section {c['chunk_id']}",
                "AI Score": c["prediction_score"],
                "Words": c["word_count"],
            }
            for c in chunks
        ]).set_index("Chunk")

        st.line_chart(timeline_df["AI Score"])
        st.caption("AI probability plotted across successive document sections (0.0 = Human, 1.0 = AI).")

        # Chunk Filter Controls
        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            filter_mode = st.radio(
                "Filter Chunks Displayed:",
                ["All Sections", "Suspicious Only (AI >= 55%)", "Human Only (AI <= 45%)"],
                horizontal=True,
            )

        filtered_chunks = chunks
        if filter_mode == "Suspicious Only (AI >= 55%)":
            filtered_chunks = [c for c in chunks if c["is_suspicious"]]
        elif filter_mode == "Human Only (AI <= 45%)":
            filtered_chunks = [c for c in chunks if c["prediction_score"] <= 0.45]

        st.markdown(f"**Displaying {len(filtered_chunks)} of {len(chunks)} sections:**")

        for c in filtered_chunks:
            is_susp = c["is_suspicious"]
            score_color = "#f87171" if is_susp else "#4ade80"
            card_class = "chunk-card-suspicious" if is_susp else "chunk-card"

            with st.expander(f"Section #{c['chunk_id']} — Likelihood: {c['prediction_score']*100:.1f}% ({c['label']}) — {c['word_count']} words"):
                sc_col1, sc_col2 = st.columns([1, 2])
                with sc_col1:
                    st.write("**Model Probabilities:**")
                    for m_name, m_val in c["model_scores"].items():
                        st.write(f"- `{m_name}`: **{m_val:.3f}**")
                    st.caption(f"Cross-model std: {c['agreement_std']:.3f}")
                with sc_col2:
                    st.write("**Passage Excerpt:**")
                    st.markdown(f"*{c['full_text']}*")


# ===========================================================================
# VIEW 3: FORENSIC EXPLAINABILITY
# ===========================================================================
elif nav_selection == "🔍 Forensic Explainability":
    st.markdown("### 🔍 Forensic Explainability & Model Agreement")
    res = st.session_state.get("last_result")

    if not res:
        st.info("Please run an analysis in the 'Document Analysis' tab first.")
    else:
        # Model Agreement Panel
        st.markdown("#### **1. Sub-Model Prediction Breakdown & Consensus**")
        st.caption("Inspect individual sub-models alongside the final calibrated ensemble:")

        m_scores = res["model_scores"].copy()
        m_scores["Ensemble Final"] = res["prediction_score"]
        
        m_df = pd.DataFrame({
            "Classifier Pipeline": list(m_scores.keys()),
            "AI Likelihood Probability": list(m_scores.values()),
        }).set_index("Classifier Pipeline")

        st.bar_chart(m_df)

        ag = res["model_agreement"]
        st.markdown(
            f"""
            <div style="background: #131c2b; border: 1px solid #28374d; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem;">
                <b>Consensus Statistics:</b> Agreement Fraction: <b>{ag['agreement_fraction']*100:.0f}%</b> | 
                Score Dispersion (Std): <b>{ag['score_std']:.3f}</b> | 
                Decision Threshold Applied: <b>{res['threshold_used']:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Lexical Contributors (Word N-Grams)
        st.markdown("#### **2. Lexical Evidence: Influential N-Grams**")
        st.caption("Top n-grams present in this document pushing the prediction toward AI vs. Human:")

        exp_words = res.get("explain_words", {})
        ai_words = exp_words.get("ai_leaning", [])
        human_words = exp_words.get("human_leaning", [])

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.markdown("##### 🔴 AI-Leaning N-Grams (+ Contribution)")
            if ai_words:
                for item in ai_words:
                    st.write(f"- `{item['term']}` — contribution: `+{item['contribution']:.3f}`")
            else:
                st.caption("No strong AI-leaning lexical n-grams detected.")

        with w_col2:
            st.markdown("##### 🟢 Human-Leaning N-Grams (- Contribution)")
            if human_words:
                for item in human_words:
                    st.write(f"- `{item['term']}` — contribution: `{item['contribution']:.3f}`")
            else:
                st.caption("No strong human-leaning lexical n-grams detected.")

        st.markdown("---")
        # Stylometric Benchmarks
        st.markdown("#### **3. Stylometric Metric Benchmark Comparison**")
        st.caption("Compares this document's stylometric indicators against training corpus medians:")

        benchmarks = res.get("stylometric_benchmarks", [])
        if benchmarks:
            bench_df = pd.DataFrame(benchmarks)
            st.dataframe(bench_df, use_container_width=True)

        # Occlusion Attribution
        st.markdown("---")
        st.markdown("#### **4. Local Feature Attribution (Leave-One-Out Occlusion)**")
        st.caption(
            "Measures the exact shift in AI probability when each structural feature is replaced "
            "with its typical training baseline value. Positive = this feature's actual value drove the prediction higher."
        )

        occlusion = res.get("explain_occlusion", [])
        if occlusion:
            occ_df = pd.DataFrame([
                {
                    "Feature": item["feature"],
                    "Probability Shift": f"{'+' if item['attribution']>=0 else ''}{item['attribution']:.3f}",
                    "Observed Value": f"{item['this_text_value']:.2f}",
                    "Corpus Median": f"{item['typical_value']:.2f}",
                }
                for item in occlusion
            ])
            st.table(occ_df)


# ===========================================================================
# VIEW 4: STYLE & PATTERN DETECTOR
# ===========================================================================
elif nav_selection == "✍️ Style & Pattern Detector":
    st.markdown("### ✍️ Suspicious Writing Patterns & Stylometric Profiling")
    res = st.session_state.get("last_result")

    if not res:
        st.info("Please run an analysis in the 'Document Analysis' tab first.")
    else:
        sp = res["suspicious_patterns"]

        sp_col1, sp_col2, sp_col3 = st.columns(3)
        with sp_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Suspicious Pattern Index</div>
                    <div class="metric-val" style="color: {'#f87171' if sp['suspicion_index']>=50 else '#4ade80'};">
                        {sp['suspicion_index']} / 100
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sp_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Sentence Burstiness</div>
                    <div class="metric-val" style="color: {'#f87171' if sp['burstiness']<0.50 else '#4ade80'};">
                        {sp['burstiness']:.2f}
                    </div>
                    <small style="color: #94a3b8;">{sp['burstiness_assessment']}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sp_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">AI Discourse Density</div>
                    <div class="metric-val" style="color: {'#f87171' if sp['discourse_marker_density']>=10 else '#60a5fa'};">
                        {sp['discourse_marker_density']:.1f}
                    </div>
                    <small style="color: #94a3b8;">per 1,000 words</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### **Forensic Pattern Observations**")
        for f in sp["flags"]:
            st.markdown(f"- 🔎 {f}")

        if sp.get("found_discourse_markers"):
            st.markdown("#### **Detected AI Discourse Phrases**")
            m_df = pd.DataFrame(sp["found_discourse_markers"], columns=["Discourse Marker", "Occurrences"])
            st.table(m_df)


# ===========================================================================
# VIEW 5: BATCH FILE PROCESSING
# ===========================================================================
elif nav_selection == "📂 Batch File Processing":
    st.markdown("### 📂 Multi-Document Batch Processing")
    st.caption("Upload multiple PDFs, DOCXs, TXTs, or a CSV file with a 'text' column for batch processing.")

    batch_files = st.file_uploader(
        "Upload Multiple Files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    if batch_files:
        st.write(f"📁 Selected **{len(batch_files)}** files for evaluation.")
        if st.button("⚡ Process All Documents", type="primary"):
            items_to_predict = []
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            for idx, bf in enumerate(batch_files):
                status_text.text(f"Extracting [{idx+1}/{len(batch_files)}]: {bf.name}...")
                try:
                    if bf.name.lower().endswith(".csv"):
                        csv_df = read_batch_csv(bf)
                        for row_i, text_val in enumerate(csv_df["text"]):
                            items_to_predict.append((f"{bf.name}#row_{row_i+1}", text_val))
                    else:
                        txt_val, _ = load_uploaded_file(bf)
                        items_to_predict.append((bf.name, txt_val))
                except Exception as ex:
                    st.error(f"Failed to read {bf.name}: {ex}")

            status_text.text(f"Running multi-model inferences across {len(items_to_predict)} documents...")
            results_df = predictor.predict_batch(items_to_predict)
            progress_bar.progress(1.0)
            status_text.text("Batch analysis complete!")
            st.session_state["batch_df"] = results_df

    batch_res = st.session_state.get("batch_df")
    if batch_res is not None:
        st.markdown("#### **Batch Forensic Results Summary**")
        st.dataframe(batch_res, use_container_width=True)

        csv_bytes = batch_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_bytes,
            file_name=f"batch_forensic_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ===========================================================================
# VIEW 6: EVALUATION & LEAKAGE CHECK
# ===========================================================================
elif nav_selection == "📊 Evaluation & Leakage Check":
    st.markdown("### 📊 Model Evaluation, Calibration & Topic Leakage Audit")
    
    metrics_path = os.path.join(config.REPORTS_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        st.markdown("#### **1. Generalization Audit: Random vs. Group-Aware Split**")
        st.caption(metrics.get("comparison", {}).get("note", ""))

        comp = metrics.get("comparison", {})
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🎲 Random Stratified Split (Standard Baseline)")
            r_data = comp.get("random_stratified", {})
            st.write(f"- Test Accuracy: **{r_data.get('test_accuracy', 0)*100:.2f}%**")
            st.write(f"- Test F1-Score: **{r_data.get('test_f1', 0):.4f}**")
            st.write(f"- Test ROC-AUC: **{r_data.get('test_roc_auc', 0):.4f}**")

        with c2:
            st.markdown("##### 🛡️ Group-Aware Split on `prompt_name` (Unseen Topics)")
            g_data = comp.get("group_aware_prompt_name", {})
            st.write(f"- Test Accuracy: **{g_data.get('test_accuracy', 0)*100:.2f}%**")
            st.write(f"- Test F1-Score: **{g_data.get('test_f1', 0):.4f}**")
            st.write(f"- Test ROC-AUC: **{g_data.get('test_roc_auc', 0):.4f}**")

        st.markdown("---")
        st.markdown("#### **2. Calibration & Error Analysis**")
        
        calib = metrics.get("calibration", {})
        err = metrics.get("error_analysis", {})
        
        e1, e2 = st.columns(2)
        with e1:
            st.write(f"**Brier Score (Calibrated Probabilities):** `{calib.get('brier_score', 0):.4f}` (lower is better)")
            st.write(f"**False Positive Rate (Human called AI):** `{err.get('false_positive_rate', 0)*100:.2f}%`")
            st.write(f"**False Negative Rate (AI called Human):** `{err.get('false_negative_rate', 0)*100:.2f}%`")

        with e2:
            st.write(f"**Evaluated Test Samples:** `{calib.get('total_samples', 0):,}`")
            st.write(f"**False Positive Instances:** `{err.get('n_false_positives', 0)}`")
            st.write(f"**False Negative Instances:** `{err.get('n_false_negatives', 0)}`")

        if metrics.get("structural_permutation_importance"):
            st.markdown("#### **3. Validation Permutation Importance (Stylometric Features)**")
            st.caption("True empirical drop in ROC-AUC when each feature is shuffled on held-out validation data:")
            st.table(metrics["structural_permutation_importance"][:8])
    else:
        st.info("No training report found. Run `python training/train.py` to generate real benchmark metrics.")


# ===========================================================================
# VIEW 7: SYSTEM & MODEL HEALTH
# ===========================================================================
elif nav_selection == "⚙️ System & Model Health":
    st.markdown("### ⚙️ System Diagnostics & Operational Health")
    health = predictor.get_model_health()

    st.markdown(f"**System Status:** <span style='color: #4ade80; font-weight: bold;'>{health['status']}</span>", unsafe_allow_html=True)
    st.markdown(f"**Model Directory:** `{health['models_dir']}`")

    st.markdown("#### **Artifact Status Matrix**")
    comp_df = pd.DataFrame([
        {"Component": k, "Loaded Successfully": "✅ YES" if v else "❌ NO"}
        for k, v in health["components"].items()
    ])
    st.table(comp_df)

    st.markdown("#### **Active Ensemble Weights**")
    st.json(health["ensemble_weights"])

    st.markdown("#### **Cache Management**")
    st.write(f"Active in-memory cached predictions: **{health['cache_entries']}**")
    if st.button("🧹 Clear In-Memory Cache"):
        predictor._cache.clear()
        st.success("Cache cleared successfully.")


# ===========================================================================
# VIEW 8: PREDICTION HISTORY
# ===========================================================================
elif nav_selection == "📜 Prediction History":
    st.markdown("### 📜 Session Prediction History")
    hist = st.session_state.get("history", [])

    if not hist:
        st.info("No documents analyzed in this session yet.")
    else:
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)

        c_down, c_clear = st.columns([1, 1])
        with c_down:
            csv_data = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Session History (CSV)",
                data=csv_data,
                file_name="session_prediction_history.csv",
                mime="text/csv",
            )
        with c_clear:
            if st.button("🗑️ Clear History Log"):
                st.session_state["history"] = []
                st.rerun()
