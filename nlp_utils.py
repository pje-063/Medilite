import pandas as pd
import re
import os
import json
from transformers import MarianMTModel, MarianTokenizer

# --- 1. SETUP & DATA LOADING ---
tr_model_name = "Helsinki-NLP/opus-mt-en-ml"
tr_tokenizer = MarianTokenizer.from_pretrained(tr_model_name)
tr_model = MarianMTModel.from_pretrained(tr_model_name)

def load_medical_glossary(csv_path='medical_glossary.csv'):
    # Get the directory where nlp_utils.py is located
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, csv_path)
    
    if os.path.exists(full_path):
        return pd.read_csv(full_path, encoding='utf-8')
    else:
        # Debug print to see where it's looking in logs
        print(f"GLOSSARY ERROR: Could not find file at {full_path}")
    return None

GLOSSARY_DF = load_medical_glossary()

# --- 2. THE CLEANING LOGIC ---
def post_process_translation(mal_text, glossary_df):
    if mal_text is None: return ""

    # 1. Tone Fixes (Stay the same)
    tone_fixes = {r'അവൾക്ക്': 'ഇവർക്ക്', r'അവൾ': 'ഇവർ', r'അവളുടെ': 'ഇവരുടെ', r'അവൻ': 'ഇദ്ദേഹം'}
    for pattern, replacement in tone_fixes.items():
        mal_text = mal_text.replace(pattern, replacement)

    # 2. Advanced Glossary Mapping
    if glossary_df is not None:
        # Sort by length (longest first) to prevent partial matching
        glossary_df['match_len'] = glossary_df['English_Term'].astype(str).str.len()
        sorted_df = glossary_df.sort_values('match_len', ascending=False)

        for _, row in sorted_df.iterrows():
            eng_phrase = str(row['English_Term']).strip()  # This is the variable name
            mal_phrase = str(row['Malayalam_Translation']).strip()

            pattern = re.compile(re.escape(eng_phrase), re.IGNORECASE)
            mal_text = pattern.sub(mal_phrase, mal_text)

    # 3. Clean 'Management Management' bug
    mal_text = re.sub(r'(\b\w+\b)( \1)+', r'\1', mal_text)
    return mal_text

# --- 3. THE MAIN WORKFLOW ---
def translate_to_malayalam(text):
    sentences = re.split(r'(?<=[.!?;]) +', text)
    output = []

    for sent in sentences:
        sent = sent.strip()
        if not sent: continue

        # STEP A: Generate Raw Translation
        inputs = tr_tokenizer(sent, return_tensors="pt", truncation=True)
        translated = tr_model.generate(**inputs, max_new_tokens=150)
        raw_malayalam = tr_tokenizer.decode(translated[0], skip_special_tokens=True)

        # STEP B: Run the Cleaning Function (The "Workflow")
        cleaned_malayalam = post_process_translation(raw_malayalam, GLOSSARY_DF)

        output.append(cleaned_malayalam)

    return " ".join(output)

def get_highlighted_summary(text, highlight_list):
    """
    Handles multi-word highlighting for the Gradio UI.
    """
    if not text: return ""
    
    # Sort: Longest phrases first to avoid partial matching
    sorted_terms = sorted(highlight_list, key=len, reverse=True)
    highlighted = text
    
    for term in sorted_terms:
        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(r'**\1**', highlighted)
        
    return highlighted