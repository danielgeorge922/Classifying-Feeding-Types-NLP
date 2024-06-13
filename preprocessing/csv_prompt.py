import pandas as pd

# Load the dataset
df = pd.read_csv('classified_clinical_notes.csv')

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Display the sentence
    print(f"Sentence: {row['notes']}")

    # Prompt for classification input
    while True:
        try:
            classification = int(input("Enter classification (0: Neither, 1: Breastfeeding, 2: Bottle feeding): "))
            if classification in [0, 1, 2]:
                break
            else:
                print("Invalid input. Please enter 0, 1, or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, or 2).")

    # Update the classification in the DataFrame
    df.at[index, 'classification'] = classification

    # Write the updated DataFrame back to the CSV file
    df.to_csv('classified_clinical_notes.csv', index=False)

print("Classification complete. Updated CSV file saved.")
