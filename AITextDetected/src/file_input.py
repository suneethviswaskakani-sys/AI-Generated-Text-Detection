"""
Advanced file input handling for the AI Text Detection System.
Supports .txt, .docx, .pdf, and .csv (batch) with robust extraction,
cleaning, header/footer deduplication, de-hyphenation, reference isolation,
and forensic extraction diagnostics.
"""
import io
import os
import re
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd


# Regex patterns for cleaning and structure reconstruction
PAGE_NUM_RE = re.compile(
    r"^\s*(?:page\s+)?(?:\[?\d+\]?)(?:\s*(?:of|/|-)\s*\d+)?\s*$",
    re.IGNORECASE
)
HYPHEN_BREAK_RE = re.compile(r"(\b\w+)-\s*[\r\n]+\s*(\w+\b)")
MULTIPLE_SPACES_RE = re.compile(r"[ \t]+")
REFERENCE_HEADER_RE = re.compile(
    r"(?:[\r\n]+|\b)(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED|LITERATURE CITED)\s*(?:\[1\]|1\.|\(?1\)?|[\r\n]+)",
    re.IGNORECASE
)


def clean_page_lines(lines: List[str]) -> List[str]:
    """Clean and filter out obvious artifacts like standalone page numbers."""
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if PAGE_NUM_RE.match(stripped):
            continue
        cleaned.append(stripped)
    return cleaned


def detect_repeated_headers_footers(page_lines_list: List[List[str]], min_fraction: float = 0.4) -> Tuple[set, set]:
    """
    Detect repeated header and footer lines that occur across multiple pages.
    Usually headers appear in the first 2-3 lines of a page, and footers in the last 2-3 lines.
    """
    n_pages = len(page_lines_list)
    if n_pages <= 2:
        return set(), set()

    header_counts: Dict[str, int] = {}
    footer_counts: Dict[str, int] = {}

    for lines in page_lines_list:
        if not lines:
            continue
        # Check first 3 lines for headers
        for line in lines[:3]:
            # Normalize whitespace and lowercase for matching
            norm = MULTIPLE_SPACES_RE.sub(" ", line).strip()
            if len(norm) > 4 and not norm.isdigit():
                header_counts[norm] = header_counts.get(norm, 0) + 1

        # Check last 3 lines for footers
        for line in lines[-3:]:
            norm = MULTIPLE_SPACES_RE.sub(" ", line).strip()
            if len(norm) > 4 and not norm.isdigit():
                footer_counts[norm] = footer_counts.get(norm, 0) + 1

    min_count = max(2, int(n_pages * min_fraction))
    detected_headers = {k for k, count in header_counts.items() if count >= min_count}
    detected_footers = {k for k, count in footer_counts.items() if count >= min_count}

    return detected_headers, detected_footers


def dehyphenate_and_normalize(text: str) -> str:
    """
    Join hyphenated words broken across line wraps (e.g. 'com- \n prehensive' -> 'comprehensive')
    and normalize line breaks into natural paragraph structure.
    """
    # Fix hyphenated wrapped words
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)
    
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs by double newlines or single newlines
    # If text uses single newlines for line wrapping, join lines within paragraphs
    paragraphs = text.split("\n\n")
    cleaned_paras = []
    for para in paragraphs:
        # Replace remaining single newlines inside paragraph with space
        p = " ".join(line.strip() for line in para.split("\n") if line.strip())
        p = MULTIPLE_SPACES_RE.sub(" ", p).strip()
        if p:
            cleaned_paras.append(p)

    return "\n\n".join(cleaned_paras)


def strip_references(text: str) -> Tuple[str, str, bool]:
    """
    Detect and isolate standard Reference / Bibliography sections to prevent citation
    blocks from skewing lexical and stylometric features.
    Returns: (cleaned_body_text, references_text, has_references)
    """
    match = REFERENCE_HEADER_RE.search(text)
    if match:
        split_pos = match.start()
        # Ensure references aren't the entire document (must have substantial text before)
        if split_pos > 200:
            body = text[:split_pos].strip()
            refs = text[split_pos:].strip()
            return body, refs, True
    return text, "", False


def read_pdf_with_diagnostics(file_obj) -> Tuple[str, Dict[str, Any]]:
    """
    Advanced PDF reader that extracts text, strips repeated headers/footers,
    removes page numbers, dehyphenates wrapped words, isolates references,
    and returns comprehensive diagnostics.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf is not installed; run `pip install pypdf`.") from e

    # Support bytes, io.BytesIO, or file paths
    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = io.BytesIO(file_obj)

    reader = PdfReader(file_obj)
    n_pages = len(reader.pages)
    
    page_texts: List[str] = []
    page_lines_list: List[List[str]] = []
    blank_pages = 0

    for idx, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            blank_pages += 1
        page_lines_list.append(lines)
        page_texts.append(raw)

    # Detect repeating headers and footers
    headers, footers = detect_repeated_headers_footers(page_lines_list)

    # Filter pages
    cleaned_pages = []
    removed_header_count = 0
    removed_footer_count = 0

    for lines in page_lines_list:
        page_filtered = []
        for line in lines:
            norm = MULTIPLE_SPACES_RE.sub(" ", line).strip()
            if norm in headers:
                removed_header_count += 1
                continue
            if norm in footers:
                removed_footer_count += 1
                continue
            if PAGE_NUM_RE.match(norm):
                continue
            page_filtered.append(line)
        cleaned_pages.append("\n".join(page_filtered))

    combined_text = "\n\n".join(p for p in cleaned_pages if p.strip())
    normalized_text = dehyphenate_and_normalize(combined_text)

    # Check for references section
    body_text, refs_text, has_refs = strip_references(normalized_text)
    
    # Calculate diagnostics
    total_words = len(body_text.split())
    total_chars = len(body_text)
    refs_words = len(refs_text.split()) if has_refs else 0

    warnings = []
    if blank_pages > 0:
        warnings.append(f"{blank_pages} of {n_pages} pages contained no extractable text (possible scanned/image content).")
    if total_words < 50:
        warnings.append("Extracted word count is extremely low (<50 words). Analysis may be unreliable.")
    if removed_header_count > 0:
        warnings.append(f"Removed {removed_header_count} repeating header instances across pages.")
    if removed_footer_count > 0:
        warnings.append(f"Removed {removed_footer_count} repeating footer instances across pages.")
    if has_refs:
        warnings.append(f"Detected and isolated bibliography/references section ({refs_words} words excluded from core scoring).")

    # Determine extraction quality
    if total_words > 200 and blank_pages == 0:
        quality = "HIGH"
    elif total_words >= 50 and (blank_pages / max(1, n_pages)) < 0.3:
        quality = "MEDIUM"
    elif total_words > 0:
        quality = "LOW"
    else:
        quality = "POOR / OCR NEEDED"

    diagnostics = {
        "file_type": "PDF",
        "page_count": n_pages,
        "blank_pages": blank_pages,
        "total_words": total_words,
        "total_chars": total_chars,
        "removed_headers_count": removed_header_count,
        "removed_footers_count": removed_footer_count,
        "detected_headers": list(headers),
        "detected_footers": list(footers),
        "references_isolated": has_refs,
        "references_word_count": refs_words,
        "extraction_quality": quality,
        "warnings": warnings,
    }

    return body_text, diagnostics


def read_docx_with_diagnostics(file_obj) -> Tuple[str, Dict[str, Any]]:
    """Read a DOCX document, extracting paragraphs and tables, with diagnostics."""
    try:
        import docx
    except ImportError as e:
        raise ImportError("python-docx is not installed; run `pip install python-docx`.") from e

    if isinstance(file_obj, (bytes, bytearray)):
        file_obj = io.BytesIO(file_obj)

    document = docx.Document(file_obj)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    
    # Also extract text from tables
    table_texts = []
    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                table_texts.append(row_text)

    combined = "\n\n".join(paragraphs)
    if table_texts:
        combined += "\n\n" + "\n".join(table_texts)

    normalized = dehyphenate_and_normalize(combined)
    body_text, refs_text, has_refs = strip_references(normalized)
    
    total_words = len(body_text.split())
    total_chars = len(body_text)

    diagnostics = {
        "file_type": "DOCX",
        "paragraph_count": len(paragraphs),
        "table_count": len(document.tables),
        "total_words": total_words,
        "total_chars": total_chars,
        "references_isolated": has_refs,
        "references_word_count": len(refs_text.split()) if has_refs else 0,
        "extraction_quality": "HIGH" if total_words > 50 else "LOW",
        "warnings": ["Low word count (<50 words)."] if total_words < 50 else [],
    }

    return body_text, diagnostics


def read_txt_with_diagnostics(file_obj) -> Tuple[str, Dict[str, Any]]:
    """Read a plain text file with multi-encoding fallback."""
    if hasattr(file_obj, "read"):
        raw_bytes = file_obj.read()
    else:
        raw_bytes = file_obj

    if isinstance(raw_bytes, str):
        raw_text = raw_bytes
    else:
        # Try UTF-8 first, then Latin-1, then CP1252
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                raw_text = raw_bytes.decode(enc)
                break
            except (UnicodeDecodeError, AttributeError):
                continue
        else:
            raw_text = raw_bytes.decode("utf-8", errors="replace")

    normalized = dehyphenate_and_normalize(raw_text)
    body_text, refs_text, has_refs = strip_references(normalized)

    total_words = len(body_text.split())
    diagnostics = {
        "file_type": "TXT",
        "total_words": total_words,
        "total_chars": len(body_text),
        "references_isolated": has_refs,
        "extraction_quality": "HIGH" if total_words > 50 else "LOW",
        "warnings": ["Low word count (<50 words)."] if total_words < 50 else [],
    }

    return body_text, diagnostics


def read_txt(file_obj) -> str:
    text, _ = read_txt_with_diagnostics(file_obj)
    return text


def read_docx(file_obj) -> str:
    text, _ = read_docx_with_diagnostics(file_obj)
    return text


def read_pdf(file_obj) -> str:
    text, _ = read_pdf_with_diagnostics(file_obj)
    return text


def read_batch_csv(file_obj, text_column: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV for batch analysis. If text_column is not given, looks for a
    column named 'text' (case-insensitive); raises clear error otherwise.
    """
    df = pd.read_csv(file_obj)
    if text_column is None:
        candidates = [c for c in df.columns if c.strip().lower() == "text"]
        if not candidates:
            raise ValueError(
                f"No 'text' column found. Available columns: {list(df.columns)}. "
                "Please rename your text column to 'text' or select it explicitly."
            )
        text_column = candidates[0]
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    df = df.rename(columns={text_column: "text"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    return df[df["text"].str.strip() != ""].reset_index(drop=True)


def load_uploaded_file(uploaded_file) -> Tuple[str, Dict[str, Any]]:
    """
    Dispatch by extension for the Streamlit single/batch file uploader.
    Returns (cleaned_text, diagnostics_dict).
    """
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return read_txt_with_diagnostics(uploaded_file)
    if name.endswith(".docx"):
        return read_docx_with_diagnostics(uploaded_file)
    if name.endswith(".pdf"):
        return read_pdf_with_diagnostics(uploaded_file)
    raise ValueError(f"Unsupported file type: {name}. Supported: .txt, .docx, .pdf")
