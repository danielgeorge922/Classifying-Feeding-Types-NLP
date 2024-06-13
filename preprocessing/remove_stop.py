import spacy
import pandas as pd

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

# Path to the CSV file
csv_file_path = 'classified_clinical_notes.csv'

# Define a function to clean the text by removing stop words
def remove_stop_words(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.text for token in doc if not token.is_stop])
    return cleaned_text

# Load the dataset
df = pd.read_csv(csv_file_path)

# Apply the function to remove stop words from each sentence
df['notes'] = df['notes'].apply(remove_stop_words)

# Save the cleaned sentences along with the classifications to a new CSV file
cleaned_csv_file_path = 'cleaned_clinical_notes.csv'
df[['cleaned_notes', 'classification']].to_csv(cleaned_csv_file_path, index=False)

print(f"Cleaned sentences with classifications saved to {cleaned_csv_file_path}")
