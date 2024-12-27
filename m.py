import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from pdfminer.high_level import extract_text
from tabula import read_pdf


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import pandas as pd
from nltk.tokenize import sent_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models and pipelines
logging.info("Initializing models...")
MODEL_NAME = "microsoft/layoutlm-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
classifier = pipeline("text-classification", model=classification_model, tokenizer=tokenizer)

summarizer = pipeline("summarization", model="t5-small")
nlp = spacy.load("en_core_web_sm")

# Functions
def extract_text_from_pdf(pdf_path):
    """Extract plain text from a PDF file."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""



def extract_tables_from_pdf(pdf_path):
    try:
        
        tables = read_pdf(pdf_path, pages="all", multiple_tables=True)
        table_data = []
        
        for i, table in enumerate(tables):
            
            table_data.append({
                "TableNumber": i + 1,
                "Data": table.to_dict(orient="list")  
            })
        
        return table_data
    except Exception as e:
        logging.error(f"Error extracting tables from {pdf_path}: {e}")
        return []


def classify_document(text):
    """Classify the document based on its content."""
    try:
        prediction = classifier(text)
        return prediction[0]['label']
    except Exception as e:
        logging.error(f"Error classifying document: {e}")
        return "Unknown"

def summarize_text(text, max_length=100, min_length=30):
    """Summarize the provided text."""
    try:
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) > max_length:
                chunks.append(chunk)
                chunk = ""
            chunk += sentence + " "
        if chunk:
            chunks.append(chunk)

        summaries = [summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
                     for chunk in chunks]
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "No summary available."

def extract_entities(text):
    """Extract named entities and specific patterns."""
    try:
        doc = nlp(text)
        entities = {
            "Names": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            "Organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "Locations": [ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"}],
            "Dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "Emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            "Monetary Values": re.findall(r'\b[$₹€]\s?\d+(?:,\d{3})*(?:\.\d{2})?\b', text),
            "Percentages": re.findall(r'\b\d+(\.\d+)?%\b', text)
        }
        return entities
    except Exception as e:
        logging.error(f"Error extracting entities: {e}")
        return {}

def process_document(filename, directory):
    """Process a single PDF document."""
    file_path = os.path.join(directory, filename)
    logging.info(f"Processing {file_path}...")

    text = extract_text_from_pdf(file_path)
    if not text.strip():
        return None

    tables = extract_tables_from_pdf(file_path)
    category = classify_document(text)
    summary = summarize_text(text)
    entities = extract_entities(text)

    logging.info(f"Processed {filename}.")
    return {
        "Filename": filename,
        "Category": category,
        "Summary": summary,
        "Tables": tables,
        "Entities": entities
    }

def process_documents(directory):
    """Process all PDF documents in the specified directory."""
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_document, filename, directory)
            for filename in os.listdir(directory) if filename.endswith(".pdf")
        ]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return results


if __name__ == "__main__":
    pdf_directory = "D:/Appian/pdfs"
    if not os.path.isdir(pdf_directory):
        logging.error(f"Directory {pdf_directory} does not exist.")
        exit(1)

    results = process_documents(pdf_directory)

    if results:
       
        flattened_results = []
        for result in results:
            flattened_results.append({
                "Filename": result["Filename"],
                "Category": result["Category"],
                "Summary": result["Summary"],
                "Names": "\n".join(result["Entities"]["Names"]),
                "Emails": "\n".join([f"email: {email}" for email in result["Entities"]["Emails"]]),
                "Organizations": ", ".join(result["Entities"]["Organizations"]),
                "Locations": ", ".join(result["Entities"]["Locations"]),
                "Monetary Values": ", ".join(result["Entities"]["Monetary Values"]),
                "Percentages": ", ".join(result["Entities"]["Percentages"]),
                "Tables Extracted": len(result["Tables"])
            })

        df = pd.DataFrame(flattened_results)
        output_file = "processed_financial_documents.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to '{output_file}'.")
    else:
        logging.warning("No results to save.")