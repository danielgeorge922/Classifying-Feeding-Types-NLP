import os
import spacy
from spellchecker import SpellChecker
import csv
from spacy.matcher import Matcher

# Initialize the spell checker
spell = SpellChecker()

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

# Path to the CSV file
csv_file_path = '../data/cleaned_data/raw_clinical_notes.csv'

# Initialize matcher and add patterns for breastfeeding

breastfeeding_matcher = Matcher(nlp.vocab)

breastfeeding_patterns = [
    [{'LOWER': 'breastfeed'}],
    [{'LOWER': 'breast'}, {'LOWER': 'feed', 'OP': '*'}],
    [{'LOWER': 'breastfeeding'}],
    [{'LOWER': 'breastfed'}],
    [{'LOWER': 'nursing'}],
    [{'LOWER': 'lactate'}],
    [{'LOWER': 'lactation'}],
    [{'LOWER': 'nipple'}],
    [{'LOWER': 'nipples'}],
    [{'LOWER': 'latch'}, {'LOWER': 'on'}],
    [{'LOWER': 'breast'}, {'LOWER': 'milk'}],
    [{'LOWER': 'directly'}, {'LOWER': 'from'}, {'LOWER': 'breast'}],
    [{'LOWER': 'suckling'}],
[{'LOWER': 'suckle'}],

    [{'LOWER': 'exclusive'}, {'LOWER': 'breastfeeding'}],
    [{'LOWER': 'breast'}, {'LOWER': 'pump'}]
]
breastfeeding_matcher.add('breastfeeding', breastfeeding_patterns)



bottlefeeding_matcher = Matcher(nlp.vocab)

# Define patterns for bottle feeding
bottlefeeding_patterns = [
    [{'LOWER': 'bottlefeeding'}],
    [{'LOWER': 'bottle'}, {'LOWER': 'fed'}],
    [{'LOWER': 'formula'}, {'LOWER': 'feeding'}],
    [{'LOWER': 'formula'}],
    [{'LOWER': 'bottle'}, {'LOWER': 'feed'}],
    [{'LOWER': 'formula'}, {'LOWER': 'milk'}],
    [{'LOWER': 'bottle'}],
    [{'LOWER': 'pump'}],
    [{'LOWER': 'formula'}, {'LOWER': 'fed'}],
    [{'LOWER': 'bottle'}, {'LOWER': 'feeds'}],
    [{'LOWER': 'infant'}, {'LOWER': 'formula'}],
    [{'LOWER': 'expressed'}, {'LOWER': 'milk'}]
]
bottlefeeding_matcher.add('bottlefeeding', bottlefeeding_patterns)

# Define the classification function
def classify(text):
    doc = nlp(text)
    found_breast_matches = breastfeeding_matcher(doc)
    found_bottle_matches = bottlefeeding_matcher(doc)

    """"
    CLASSIFCATION
    =============
    0: Neither related to breast feeding or bottle feeding
    1: Note deals with breast feeding
    2: Note deals with bottle feeding
    3: Note deels with both breast and bottle feeding
    """""
    if found_breast_matches and found_bottle_matches:
        return 3  # Both breastfeeding and bottle feeding
    elif found_breast_matches:
        return 1  # Breastfeeding
    elif found_bottle_matches:
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
output_csv_file_path = '../data/cleaned_data/classified_clinical_notes.csv'
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
