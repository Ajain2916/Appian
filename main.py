import pytesseract
import cv2
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

# Step 1: Text Extraction from Images
def extract_text_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(image)
    return text

# Step 2: Named Entity Recognition (NER) for Person Association
def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GOV_ID": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["PERSON"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["ORG"].append(ent.text)
        elif ent.label_ in ["ID", "GPE"]:  # Example for ID/Government IDs
            entities["GOV_ID"].append(ent.text)
    return entities

# Step 3: Document Classification
# Dummy dataset for training
data = [
    {"text": "This is a pay stub from XYZ Corp.", "category": "Pay Stub"},
    {"text": "This is a bank statement from ABC Bank.", "category": "Bank Statement"},
    {"text": "This is an ID card issued by the government.", "category": "Identity Document"},
    {"text": "This is a tax return form for the year 2023.", "category": "Tax Document"}
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Vectorize text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["category"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Evaluate Classifier
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 4: Integration
def classify_document(text):
    processed_text = vectorizer.transform([text])
    prediction = classifier.predict(processed_text)
    return prediction[0]

# Example Workflow
if __name__ == "__main__":
    # Extract text from an image
    image_path = "D:/Appian/WhatsApp Image 2024-12-20 at 23.03.58_0091005c.jpg"  
    extracted_text = extract_text_from_image(image_path)
    print("Extracted Text:", extracted_text)
    
    # Extract entities
    entities = extract_entities(extracted_text)
    print("Extracted Entities:", entities)
    
    # Classify document
    document_category = classify_document(extracted_text)
    print("Document Category:", document_category)
