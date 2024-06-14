import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

df = pd.read_csv('../data/cleaned_data/final_notes.csv')

# Split the dataset into features and target
X = df['notes']
y = df['classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_vect = vectorizer.fit_transform(X_train).toarray()
X_test_vect = vectorizer.transform(X_test).toarray()

# Train a simple model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred))
