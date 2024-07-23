# Classifying-Feeding-Types-NLP

This project involves classifying feeding types using Natural Language Processing (NLP) techniques with Logistic Regression and Random Forest models.

## Directory Structure

![image](https://github.com/user-attachments/assets/57b03b12-2ea2-46a3-b00c-326414c2454f)


## Data Folder

This folder contains the cleaned data used for training and testing the models.

### cleaned_data

Inside this folder is the cleaned CSV file (`final_notes.csv`) containing the data for classification.

## Model Folder

This folder contains the scripts for training and evaluating the models.

### logistic_regression_model.py

This script preprocesses the data, vectorizes the text data using TF-IDF, and trains a Logistic Regression model with hyperparameter tuning. The preprocessing steps include:

- Lowercasing the text data.
- Removing punctuation.
- Removing stopwords using NLTK's stopwords list.
- Stemming the words using PorterStemmer.

### random_forest_model.py

This script preprocesses the data, vectorizes the text data using TF-IDF, and trains a Random Forest model with hyperparameter tuning. The preprocessing steps include:

- Lowercasing the text data.
- Removing punctuation.
- Removing stopwords using NLTK's stopwords list.
- Lemmatizing the words using WordNetLemmatizer.

## Preprocessing Folder

This folder contains scripts for preprocessing the data before training the models.

### classify_notes.py

Contains the script for classifying notes.

### convert_notes_to_csv.py

Contains the script for converting notes to CSV format.

### csv_prompt.py

Contains the script for prompting CSV conversion.

### explore_classified_notes.ipynb

A Jupyter notebook for exploring classified notes.

### machine_learning_basics.py

Contains basic machine learning scripts and utilities.

## Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Results from running with Random Forest Classifier

![image](https://github.com/user-attachments/assets/3c70b229-05a5-45b6-b782-71262f48aac7)

