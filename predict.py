# predict.py
import joblib

# Load the model and vectorizer
model = joblib.load('api_call_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def classify_request(request):
    request_vectorized = vectorizer.transform([request])
    prediction = model.predict(request_vectorized)[0]
    return "Requires API call." if prediction == 1 else "Does not require API call."

# Example usage
if __name__ == "__main__":
    request = input("Enter your user request: ")
    print(classify_request(request))