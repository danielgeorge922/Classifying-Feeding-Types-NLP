import os
import csv

# Directory containing the clinical notes
directory = '../data/clinical_notes'

# List to store the content of each file
notes = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # Ensure we only process .txt files
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            content = file.read().strip()
            notes.append(content)

# Specify the output CSV file
output_csv = '../data/cleaned_data/raw_clinical_notes.csv'

# Write the notes to the CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['notes'])
    # Write the notes
    for note in notes:
        csvwriter.writerow([note])

print(f'Successfully saved to {output_csv}')
