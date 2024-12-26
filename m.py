import os
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import pandas as pd



model_name = "microsoft/layoutlm-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


nlp = spacy.load("en_core_web_sm")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def classify_document(text):
    try:
        prediction = classifier(text)
        return prediction[0]['label']
    except Exception as e:
        print(f"Error classifying document: {e}")
        return "Unknown"


def summarize_text(text, max_length=100, min_length=30):
    
    summarizer = pipeline("summarization", model="t5-small")
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return "No summary available."


def extract_entities(text):
    try:
        doc = nlp(text)
        entities = {
            "Names": [],
            "Government IDs": [],
            "Emails": []
        }
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["Names"].append(ent.text)
            elif ent.label_ == "GPE": 
                entities["Government IDs"].append(ent.text)
            elif ent.label_ == "EMAIL":
                entities["Emails"].append(ent.text)
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return {}


def process_documents(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")

            text = extract_text_from_pdf(file_path)

            category = classify_document(text)

            summary = summarize_text(text)

            entities = extract_entities(text)

            results.append({
                "Filename": filename,
                "Category": category,
                "Summary": summary,
                "Entities": entities
            })

            print(f"Processed {filename}.")
    return results

if __name__ == "__main__":

    pdf_directory = "D:/Appian/pdfs" 

    results = process_documents(pdf_directory)

    for result in results:
        print("\n--- Result ---")
        print(f"Filename: {result['Filename']}")
        print(f"Category: {result['Category']}")
        print(f"Summary: {result['Summary']}")
        print(f"Entities: {result['Entities']}")

 
    df = pd.DataFrame(results)
    df.to_csv("processed_documents.csv", index=False)
    print("\nResults saved to 'processed_documents.csv'.")
