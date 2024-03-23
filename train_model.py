# train_model.py
import joblib
import json  # Added to handle JSON format.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
with open('data.json') as file:
    data = json.load(file)

X = [item['text'] for item in data]
y = [item['label'] for item in data]

# Text vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, predictions))

# Save the model and vectorizer
joblib.dump(model, 'api_call_classifier_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
