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

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Import necessary libraries
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
df = pd.read_csv('../data/cleaned_data/final_notes.csv')

# Display class distribution
print(df['classification'].value_counts())

# Preprocessing steps
df['notes'] = df['notes'].str.lower()  # Lowercasing
df['notes'] = df['notes'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation
stop = set(stopwords.words('english'))
df['notes'] = df['notes'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))  # Remove stopwords
stemmer = PorterStemmer()
df['notes'] = df['notes'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))  # Stemming

# Split the dataset into features and target
X = df['notes']
y = df['classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

# Train a model with hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
modelLR = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5, scoring='f1_weighted')
modelLR.fit(X_train_vect, y_train)

# Evaluate the model
y_pred = modelLR.predict(X_test_vect)
print(classification_report(y_test, y_pred))

# Display the best parameters
print(f"Best parameters: {modelLR.best_params_}")
