# Medical Keywords for OCR Validation
medical_keywords = [
    "patient", "history", "diagnosis", "treatment", "medication", 
    "symptoms", "fever", "pain", "diabetes", "hypertension", 
    "clinical", "blood", "cough", "prescribed", "weakness", "tablet",
    "dental", "orthopedic", "spine", "medicine", "health","chronic", "acute", "report", 
    "clinic", "dosage", "allergic", "surgery", "results"
]

# Terms to be highlighted in the final summary
HIGHLIGHT_TERMS = [
    "hypertension", "diabetes", "mellitus", "angina", "ischemia", "chronic", 
    "asthma", "bronchitis", "pneumonia", "gastritis", "gerd", "stroke", 
    "infarction", "myocardial", "migraine", "epilepsy", "neuropathy",
    "arthritis", "anemia", "cancer", "tumor", "cyst", "ulcer", "type 2 diabetes",
    "appendicitis", "multiple sclerosis", "sitagliptin", "osteoporosis", "fever", 
    "cough", "fatigue", "vertigo", "dizziness", "double vision", "headache", 
    "nausea", "vomiting", "tingling", "shortness of breath", "chest pain", 
    "abdominal", "drowsiness", "blurred vision", "stiffness", "weakness", 
    "lethargy", "localized", "febrile", "numbness", "wheezing", "prednisolone", 
    "edema", "antibiotics", "aspirin", "beta-blockers", "insulin", 
    "antihypertensive", "hypoglycemic", "analgesics", "ibuprofen", "amoxicillin", 
    "metformin", "atorvastatin", "physiotherapy", "surgery", "epley", 
    "maneuver", "statin", "steroids", "nerve block", "erenumab", "botox", 
    "pantoprazole", "gabapentin", "levetiracetam", "amlodipine", "saline", 
    "rednisolone", "intravenous", "iv", "coronary", "hypoglycemia", "hyperglycemia", 
    "tachycardia", "bradycardia", "arrhythmia", "bp", "glycemic", "hemoglobin", 
    "cholesterol", "white blood cell", "wbc", "platelets", "bilirubin", 
    "creatinine", "urea", "corticosteroids", "leukotriene", "hba1c", "neurology", 
    "cardiology", "oncology", "orthopedic", "vestibular", "spine", "lumbar", 
    "cervical", "thoracic", "bilateral", "umbilicus", "buttock", "sacral", 
    "ventricular", "pulmonary", "hepatic", "renal", "quadrant", "extremity", 
    "retinopathy", "severe", "negative", "hypoglucosemic", "follow-up", "radiology", 
    "x-ray", "mri", "numbness", "LABA", "fractures", "sinusitis", 
    "allergic", "rhinitis", "laparoscopy", "appendectomy", "lumbar puncture", 
    "intubation", "neurosurgery", "biopsy", "echocardiogram", 
    "electrocardiogram", "tonsils", "cardiac", "appetite", "endoscopy", "auscultation", 
    "staphylococcus", "streptococcus", "tuberculosis", "hiv", "hepatitis", 
    "malaria", "dengue", "neonatal", "congenital", "dementia", "alzheimer's", 
    "gerontology", "cautery", "stent", "catheter", "ventilator", "pacemaker", 
    "prosthesis", "troponin", "amylase", "lipase", "intramuscular", 
    "sublingual", "psoriasis", "eczema", "lupus", "rheumatoid", "osteoarthritis", 
    "glaucoma", "cataract", "melanoma", "lymphoma", "leukemia", "cirrhosis", 
    "hepatoma", "diverticulitis", "pancreatitis", "cholecystitis", "nephritis", 
    "urolithiasis", "pneumothorax", "emphysema", "copd", "fibrosis", 
    "sarcoidosis", "meningitis", "parkinson", "alzheimer", "sciatica", "hernia", 
    "ketoacidosis", "cyanosis", "pallor", "pruritus", "dyspnea", "emesis",
    # --- New Essential Additions ---
    "preeclampsia", "gestational", "ectopic", "placenta", "obstetrics",
    "gynecology", "hypothyroidism", "hyperthyroidism", "goiter", "syncope",
    "anaphylaxis", "haemorrhage", "hematoma", "trauma", "laceration",
    "concussion", "septicemia", "vasculitis", "ischemic", "necrosis",
    "hypoxia", "triage", "prognosis", "remission", "palliative"
]
def is_medical_document(text):
    """Validates if the OCR text is actually a medical report."""
    count = sum(1 for word in medical_keywords if word.lower() in text.lower())
    return count >= 3 # Requires at least 3 medical terms to proceed