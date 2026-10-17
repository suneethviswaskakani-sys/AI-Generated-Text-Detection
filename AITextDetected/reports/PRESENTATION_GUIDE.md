# VeriText 2.0: University Project Defense & Viva Guide
## Complete Presentation Walkthrough for Project Guide & Evaluation Committee

**Project Title**: Detection of AI-Generated Text Using a Hybrid Explainable Ensemble of Lexical, Character-Level, and Stylometric Models (Version 2.0)  
**Domain**: Artificial Intelligence, Natural Language Processing & Explainable AI (XAI)  
**Target Audience**: Project Guide, External Examiners, Department Committee  

---

## 1. 2-Minute Elevator Pitch (Opening Statement)

> *"Good morning, respected guide and members of the evaluation committee.*
>
> *Today, generative AI tools produce essays, research articles, and reports that closely mimic human writing. However, existing AI text detectors suffer from three fatal flaws: first, they are opaque 'black boxes' that give an unexplained percentage score with zero legal or academic evidence; second, they fail completely on real-world multi-page PDFs because long documents distort their statistical features; and third, they collapse when faced with mixed authorship—where a student writes the introduction but generates the body paragraphs using an LLM.*
>
> *Our project, **VeriText 2.0**, solves these challenges. We have developed an industrial-grade AI-generated text detection and forensic explainability system. It integrates a **sentence-aligned sliding window chunking engine**, an **AUC-weighted multi-model ensemble** (Word N-Gram Logistic Regression, Character N-Gram Calibrated Linear SVM, and a 27-dimensional Stylometric Gradient Boosting classifier), a **4-tier uncertainty framework**, and **local feature occlusion attribution** that proves to examiners exactly which words and stylistic markers drove the decision.*
>
> *Our system is evaluated using strict **topic-leakage control** on over 44,000 real-world samples, achieving **97.4% accuracy on completely unseen essay topics** and **100% test pass rate across 46 automated integration tests**."*

---

## 2. Five Key Technical Innovations to Highlight

When presenting slides or demonstrating the system to your guide, emphasize these five distinct engineering achievements:

### 1. Robust Document Ingestion & Forensic Structure Recovery
- **The Problem**: PDFs contain page headers, footers, line-wrap hyphens, and multi-page bibliographies that severely corrupt tokenization.
- **Our Solution**: Developed frequency-based header/footer deduplication, line-wrap de-hyphenation, and automatic reference section isolation.

### 2. Sentence-Aligned Sliding Window Chunking
- **The Problem**: Monolithic feature extraction on a 5,000-word PDF pushes length features $+30\sigma$ outside the training distribution, causing false negatives.
- **Our Solution**: Automatically partitions long documents into 280-word windows with 50-word overlaps along sentence boundaries, scoring each section independently.

### 3. Mixed Authorship Identification
- **The Problem**: Binary detectors cannot handle partially AI-written documents.
- **Our Solution**: Analyzes inter-chunk probability variance ($\text{Var} \ge 0.045$) and stylistic transitions to detect hybrid human-AI documents with high confidence.

### 4. Transparent Explainable AI (XAI) Dashboard
- **The Problem**: Teachers and editors cannot act on an opaque number without evidence.
- **Our Solution**: Provides:
  - Lexical n-gram contributions (positive vs. negative token coefficients).
  - Empirical stylometric benchmarks (burstiness, lexical diversity, readability).
  - Leave-one-out feature occlusion attribution.
  - Natural language analytical summary narratives.

### 5. Leakage-Controlled Scientific Validation
- **The Problem**: Many college projects fabricate 99.9% accuracy by testing on the same prompts used during training (topic leakage).
- **Our Solution**: Evaluated using **Group-Aware Split on `prompt_name`**, guaranteeing that test essays are on completely unseen subjects.

---

## 3. Anticipated Viva / Examiner Questions & Winning Answers

### Q1: *"Why did you use an ensemble of traditional ML models instead of just using GPT-4 or a massive transformer?"*
**Your Answer**:
> *"Deploying large foundation models for real-time detection has major drawbacks: they require expensive GPUs, introduce high API latency, lack deterministic reproducibility, and often produce circular reasoning. Furthermore, research demonstrates that transformers can be fooled by simple stylistic prompts.
> 
> Our hybrid ensemble combines sub-linear TF-IDF, character n-grams, and 27 engineered stylometric features with an optional lightweight DistilBERT branch. This runs in under **400 milliseconds on a standard laptop CPU**, consumes minimal memory, provides **exact mathematical attribution via occlusion and logistic coefficients**, and achieves **97.4% accuracy** on unseen topics without any paid API dependencies."*

---

### Q2: *"What is topic leakage, and how did you prevent it?"*
**Your Answer**:
> *"Topic leakage occurs when training and test sets contain essays written on the same prompt (e.g. 'Should driverless cars be banned?'). In that case, an n-gram model simply memorizes that 'driverless' or 'autonomous' is an AI topic, rather than learning the synthetic cadence of AI writing.
> 
> We prevented this by implementing two evaluation protocols: a standard random stratified split, and a **Group-Aware Split on `prompt_name`**. The group-aware split holds entire prompt categories out of training. Our test results prove that even when evaluating topics the models have never seen before, our ensemble retains **97.4% accuracy and 0.9961 ROC-AUC**."*

---

### Q3: *"How does the system handle mixed documents where a human writer copy-pastes AI text into their essay?"*
**Your Answer**:
> *"Single-pass detectors fail here because the human text dilutes the AI signal. Our system uses a **Sentence-Aligned Sliding Window Chunking Engine**. It breaks the document into 280-word windows and evaluates each chunk independently.
> 
> We then calculate the statistical variance across chunk scores. If the variance exceeds our threshold and contains both strongly human chunks ($\le 0.40$) and strongly AI chunks ($\ge 0.60$), the system triggers the **MIXED CHARACTERISTICS** classification and highlights the exact suspicious sections on an interactive timeline."*

---

### Q4: *"What is 'Burstiness', and why is it important in AI detection?"*
**Your Answer**:
> *"Burstiness measures the variation in sentence length and syntactic structure throughout a document. Human writers naturally vary their rhythm—combining a short, punchy sentence with a longer, complex sentence. Large Language Models, due to their statistical token prediction, generate sentences with very uniform lengths and predictable cadences.
> 
> We compute burstiness as the standard deviation of sentence lengths divided by the mean sentence length. In our empirical benchmarks, human essays average **0.62**, while AI text clusters tightly around **0.42**."*

---

### Q5: *"Are the probability scores real, or did you hardcode or fabricate them?"*
**Your Answer**:
> *"Every single probability score is computed in real time from the trained pipeline artifacts. We calibrated our Linear SVM using Sigmoid cross-validation calibration, and our ensemble achieves a **Brier score of 0.0124** on held-out test data. Nothing is hardcoded or fabricated; examiners can inspect our `training_metrics.json` report and run `python run_tests.py` to verify all 46 automated tests directly."*

---

## 4. Live Demonstration Step-by-Step Script

1. **Launch**: Open terminal and run `streamlit run app.py`.
2. **Preset 1 (Human Text)**: Click `🧑 Human Text` in the sidebar and press `Analyze Document`. Show the committee the emerald green `HUMAN-WRITTEN` badge, low confidence meter ($<20\%$), natural burstiness ($>0.60$), and human n-grams.
3. **Preset 2 (AI Text)**: Click `🤖 AI Sample` and press `Analyze Document`. Point out the red `AI-GENERATED` badge, high confidence meter ($>85\%$), discourse markers, and synthetic cadence.
4. **Preset 3 (Mixed Authorship)**: Click `🔀 Mixed Authorship Sample`. Switch to the **🧩 Chunk-Level Inspector** tab. Show how Section 1 is Human, Section 2 is AI, and Section 3 is Human, triggering the amber `MIXED CHARACTERISTICS` badge.
5. **Real PDF Ingestion**: Upload `AI_Text_Detection_Base_Paper.pdf`. Point out the **PDF Extraction Diagnostics** box (9 pages, headers removed, references isolated, 18 chunks created, high extraction quality).
6. **Explainability**: Switch to the **🔍 Forensic Explainability** tab. Show the sub-model agreement bar chart, top AI vs Human n-grams, and the leave-one-out occlusion attribution table.
7. **Batch Analysis**: Switch to **📂 Batch File Processing** and show the CSV export functionality.
8. **Evaluation Audit**: Switch to **📊 Evaluation & Leakage Check** and explain the group-aware leakage check to demonstrate academic rigor.
