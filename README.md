# Phishing Email Detection System

## Project Overview

This project implements a machine learning-based phishing email detection system. It uses Natural Language Processing (NLP) and TF-IDF vectorization to classify emails as either phishing or legitimate. Two models are trained and evaluated: Naive Bayes and Logistic Regression.

**Author:** Khatouri Zayd (2211551214)

## Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd PhshingEmails
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download NLTK data for stopwords:
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```

## How to Train

Run the training script to preprocess data, train models, and save them:

```bash
python main.py
```

This will:
- Load the phishing email dataset from `data/Phishing_Email.csv`
- Preprocess emails (lowercase, HTML stripping, stop-word removal, etc.)
- Split data into training/test sets
- Train Naive Bayes and Logistic Regression models
- Evaluate models and display performance metrics
- Save trained models to `model/` directory for use by the Flask app

## How to Run the Web App

Start the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

Enter email text in the form to get a classification (Phishing or Legitimate) with confidence score.

## Project Structure

```
PhshingEmails/
├── main.py              # Training script
├── app.py               # Flask web application
├── utils.py             # Shared preprocessing function
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/
│   └── Phishing_Email.csv  # Training dataset
├── model/
│   ├── nb_model.pkl     # Trained Naive Bayes model
│   └── vectorizer.pkl   # TF-IDF vectorizer
└── Templates/
    └── index.html       # Web interface
```
