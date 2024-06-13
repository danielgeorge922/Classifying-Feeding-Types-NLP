import os
import spacy
from spellchecker import Spellchecker
import csv
from spacy.matcher import Matcher

# Initialize the spell checker
spell = Spellchecker()

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

# Path to the CSV file
csv_file_path = 'clinical_notes.csv'

# Initialize matcher and add patterns for breastfeeding
matcher = Matcher(nlp.vocab)
breastfeeding_pattern1 = [{'LOWER': 'breastfeed'}]
breastfeeding_pattern2 = [{'LOWER': 'breast'}, {'LOWER': 'feed', 'OP': '*'}]
matcher.add('breastfeeding', [breastfeeding_pattern1, breastfeeding_pattern2])

# Define the classification function
def classify(text):
    doc = nlp(text)
    found_matches = matcher(doc)
    if found_matches:
        return 1  # Breastfeeding
    elif 'bottle' in text:
        return 2  # Bottle feeding
    else:
        return 0  # None of the above

# Define the cleaning function
def cleandata(text):
    doc = nlp(text)
    newtext = ""
    for token in doc:
        if not token.is_stop:
            # Correct the spelling of the token
            corrected_word = spell.correction(token.text)
            if corrected_word:  # Check if the corrected word is not None
                # Lemmatize the corrected word
                corrected_doc = nlp(corrected_word)
                lemmatized_word = corrected_doc[0].lemma_
                newtext += lemmatized_word + " "
    newtext = newtext.strip()
    return newtext

# Prepare to write the results to a new CSV file
output_csv_file_path = 'classified_clinical_notes.csv'
with open(output_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile_out:
    csvwriter = csv.writer(csvfile_out)
    csvwriter.writerow(['notes', 'classification'])  # Header row

    # Open the input CSV file
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile_in:
        csvreader = csv.reader(csvfile_in)

        # Skip the header row
        header = next(csvreader)
        print(f"Header: {header}")

        # Process each row
        for row in csvreader:
            if row and row[0].strip():  # Check if the row is not empty and the first column is not empty
                sentence = row[0]  # First column: Sentence
                cleaned_text = cleandata(sentence)
                classification = classify(cleaned_text)
                csvwriter.writerow([sentence, classification])

print(f"Classification results saved to {output_csv_file_path}")
