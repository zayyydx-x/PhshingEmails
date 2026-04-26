from flask import Flask, render_template, request
import pickle, os
from utils import preprocess

app = Flask(__name__, template_folder='Templates')

# Load saved model and vectorizer at startup
model = None
vectorizer = None

try:
    with open('model/nb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("WARNING: Model files not found. Please run main.py first to train the model.")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    email_text = ''
    confidence = None
    error = None

    if request.method == 'POST':
        if model is None or vectorizer is None:
            error = "Model not loaded. Please ensure model files exist in model/ folder."
        else:
            email_text = request.form.get('email_text', '').strip()
            if not email_text:
                error = "Please enter some email text to analyze."
            else:
                cleaned = preprocess(email_text)
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = round(max(proba) * 100, 1)
                result = 'PHISHING' if prediction == 1 else 'LEGITIMATE'

    return render_template('index.html',
                           result=result,
                           email_text=email_text,
                           confidence=confidence,
                           error=error)

if __name__ == '__main__':
    app.run(debug=True)