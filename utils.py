import re
import cv2
import numpy as np
from medical_domain import medical_keywords

# =========================
# IMAGE PRE-PROCESSING
# =========================
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# =========================
# CLEANING & FORMATTING
# =========================
def clean_text(text):
    """Basic whitespace cleanup."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def strict_clean_ocr(text):
    """Fixes common OCR misreads for medical units and numbers."""
    corrections = {
        "O2": "O₂",
        "mm Hg": "mmHg",
        "g dL": "g/dL",
        "cells 1L": "cells/µL"
    }
    for k, v in corrections.items():
        text = text.replace(k, v)
    
    # Remove junk characters but keep medical symbols
    text = re.sub(r'[^\w\s\.\,\:\%\-\°\/]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def bold_medical_terms(text, terms_to_bold):
    """Wraps medical terms in Markdown bold tags for the Gradio UI."""
    for term in terms_to_bold:
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        text = pattern.sub(r'**\1**', text)
    return text

# =========================
# MEDICAL DETECTION
# =========================
def is_medical_text(text):
    """Gatekeeper: Checks if the document is actually medical."""
    text = text.lower()
    primary_identifiers = ["patient", "clinical", "hospital", "doctor", "medical", "history"]
    
    has_primary = any(id_word in text for id_word in primary_identifiers)
    score = sum(1 for w in medical_keywords if w.lower() in text)

    # Valid if it has a main keyword OR at least 3 specific medical terms
    return has_primary or score >= 3