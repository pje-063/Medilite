import gradio as gr
import pytesseract
from PIL import Image
import torch
import re
import os
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from gtts import gTTS

# --- Modular Imports ---
# Ensure these files are in the same folder as app.py
from utils import clean_text, strict_clean_ocr, is_medical_text
from nlp_utils import translate_to_malayalam, get_highlighted_summary 
from medical_domain import HIGHLIGHT_TERMS 

# =========================
# SETUP & MODELS
# =========================
# Update this path if your Tesseract installation is in a different folder
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

summarizer_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(DEVICE)
summarizer_tokenizer = T5TokenizerFast.from_pretrained("t5-base", model_max_length=512)

# =========================
# MAIN PIPELINE
# =========================
def process_image(image, progress=gr.Progress()):
    if image is None: 
        return "", "Please upload an image.", "Please upload an image.", None
    
    # --- 1. OCR Extraction ---
    progress(0.2, desc="Extracting Text...")
    raw_text = pytesseract.image_to_string(image.convert("L"), config="--psm 6")
    
    # Clean text (Handles clinical units like g/dL, blood pressure formats, etc.)
    cleaned = strict_clean_ocr(clean_text(raw_text))[:1200]

    # --- 2. Domain Validation ---
    if not is_medical_text(cleaned):
        return cleaned, "### ⚠️ ERROR\nINVALID MEDICAL REPORT DETECTED.\nPlease upload a valid clinical report.", "", None

    # --- 3. Summarization (Memory Efficient) ---
    progress(0.5, desc="Analyzing Report...")
    input_text = "summarize: " + cleaned.strip()
    
    inputs = summarizer_tokenizer(
        input_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = summarizer_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,    
            min_length=50,         
            num_beams=6,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,                   
            length_penalty=1.2,    
            early_stopping=True    
        )
    
    raw_summary = summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Capitalize sentences for professionalism
    raw_summary = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), raw_summary)
    
    # --- 4. Highlighting (English UI Only) ---
    progress(0.7, desc="Processing Highlights...")
    # Using the new multi-word phrase logic from nlp_utils
    highlighted_eng = get_highlighted_summary(raw_summary, HIGHLIGHT_TERMS)


    # --- 5. Translation (Clean Input) ---
    progress(0.9, desc="Translating to Malayalam...")
    mal_summary = translate_to_malayalam(raw_summary)

    mal_summary = mal_summary.strip()
    # Find the last occurrence of Malayalam punctuation (।) or English (.)
    last_punc = max(mal_summary.rfind('।'), mal_summary.rfind('.'))
    
    if last_punc != -1:
        # Keep everything up to and including the last full stop
        mal_summary = mal_summary[:last_punc + 1]
    # ==========================================
    # --- 6. Audio Generation ---
    audio_file = "output.mp3"
    try:
        # Generate audio from the Malayalam translation
        tts = gTTS(text=mal_summary, lang="ml")
        tts.save(audio_file)
        audio_output = audio_file
    except Exception as e:
        print(f"Audio Error: {e}")
        audio_output = None

    return cleaned, highlighted_eng, mal_summary, audio_output

# =========================
# UI & CSS
# =========================
custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #f0f4f8}

/* Hides all timers and 'Preparing Output' status text */
.timer, .meta-text, .meta-text-item, .status-tracker, .progress-text, .progress-container, .loading {
    display: none !important;
    visibility: hidden !important;
}

.output-box {
    border-radius: 10px; padding: 10px; background: white; 
    margin-bottom: 10px; border: 1px solid #d1d9e0;
}
#medical-header {text-align: center; color: #004d99; font-family: 'Segoe UI', sans-serif; margin-bottom: 20px;}
"""

with gr.Blocks(title="MEDILITE") as demo:
    gr.HTML("<div id='medical-header'><h1>🩺 MEDILITE - Medical Report Summarizer</h1></div>")
    
    with gr.Tabs():
        with gr.Tab("📋 Patient Summary"):
            with gr.Row():
                
                # LEFT COLUMN: Upload
                with gr.Column(scale=1):
                    gr.Markdown("### Upload Medical Image")
                    input_img = gr.Image(type="pil", label=None)
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        submit_btn = gr.Button("Submit", variant="primary")
                
                # RIGHT COLUMN: Boxed UI results
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="output-box"):
                        gr.Markdown("#### English Summary")
                        eng_display = gr.Markdown(value="*Results will appear here...*", container=True)
                    # Malayalam Section
                    with gr.Group(elem_classes="output-box"):
                        gr.Markdown("#### മലയാളം സംഗ്രഹം (Malayalam Summary)")
                        mal_display = gr.Markdown(value="*ഫലങ്ങൾ ഇവിടെ ദൃശ്യമാകും...*", container=True)
                    
                    # Audio component without internal container
                    with gr.Group(elem_classes="output-box"):
                        gr.Markdown("#### Malayalam Audio")
                        audio_out = gr.Audio(label=None, type="filepath", container=True)

        # Technical Tab
        with gr.Tab("⚙️ OCR Extraction"):
            gr.Markdown("#### Raw OCR Extraction")
            extracted_text = gr.Textbox(lines=20, interactive=False)

    # --- Button Actions ---
    submit_btn.click(
        fn=process_image, 
        inputs=input_img, 
        outputs=[extracted_text, eng_display, mal_display, audio_out]
    )
    
    clear_btn.click(
        lambda: [None, "", "*Summary will appear here...*", "*മലയാളം സംഗ്രഹം ഇവിടെ കാണാം...*", None], 
        outputs=[input_img, extracted_text, eng_display, mal_display, audio_out]
    )

if __name__ == "__main__":
    # Final launch command
    demo.launch(css=custom_css)