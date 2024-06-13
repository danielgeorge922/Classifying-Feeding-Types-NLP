import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('cleaned_clinical_notes.csv')

# # Display the first 5 rows of the dataframe
# print(df.head())
#
# # Detect missing values
# print(df.isnull().sum())
#
# # Display unique classifications
# print(df['classification'].unique())

# Define features and labels
X = df['cleaned_notes']
y = df['classification']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

# Create a pipeline with TfidfVectorizer and LinearSVC
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Train the classifier
text_clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = text_clf.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
