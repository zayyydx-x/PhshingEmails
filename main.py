import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import pickle, os

print("=" * 60)
print("  PHISHING EMAIL DETECTION SYSTEM")
print("  Khatouri Zayd | 2211551214")
print("=" * 60)

# STEP 1: LOAD
data = pd.read_csv("data/Phishing_Email.csv", encoding='latin-1')
data = data[['Email Text', 'Email Type']].dropna()
print(f"\n[1] Dataset: {len(data)} emails | Phishing: {(data['Email Type']=='Phishing Email').sum()} | Safe: {(data['Email Type']=='Safe Email').sum()}")

# STEP 2: PREPROCESS
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['clean_text'] = data['Email Text'].apply(preprocess)
data['label'] = data['Email Type'].map({'Phishing Email': 1, 'Safe Email': 0})
print("[2] Preprocessing done")

# STEP 3: SPLIT
X = data['clean_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[3] Split: {len(X_train)} train / {len(X_test)} test")

# STEP 4: TF-IDF
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"[4] TF-IDF: {X_train_tfidf.shape[1]} features")

# STEP 5: TRAIN
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
print("[5] Models trained")

# STEP 6: PREDICT
nb_pred = nb_model.predict(X_test_tfidf)
lr_pred = lr_model.predict(X_test_tfidf)

# STEP 7: EVALUATE
def show_results(name, y_true, y_pred):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Safe Email','Phishing Email'])}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  [[TN={cm[0][0]}  FP={cm[0][1]}]")
    print(f"   [FN={cm[1][0]}  TP={cm[1][1]}]]")

show_results("NAIVE BAYES", y_test, nb_pred)
show_results("LOGISTIC REGRESSION (Comparison)", y_test, lr_pred)

# STEP 8: COMPARISON TABLE
print(f"\n{'='*60}\n  MODEL COMPARISON\n{'='*60}")
print(f"  {'Metric':<12} {'Naive Bayes':>14} {'Logistic Reg':>14}")
print(f"  {'-'*40}")
for metric, fn in [('Accuracy',accuracy_score),('Precision',precision_score),('Recall',recall_score),('F1-Score',f1_score)]:
    args = {} if metric=='Accuracy' else {}
    nb_s = fn(y_test, nb_pred) if metric=='Accuracy' else fn(y_test, nb_pred)
    lr_s = fn(y_test, lr_pred) if metric=='Accuracy' else fn(y_test, lr_pred)
    print(f"  {metric:<12} {nb_s:>14.4f} {lr_s:>14.4f}")

# STEP 9: SAVE MODELS
os.makedirs('model', exist_ok=True)
with open('model/nb_model.pkl','wb') as f: pickle.dump(nb_model, f)
with open('model/vectorizer.pkl','wb') as f: pickle.dump(vectorizer, f)
print(f"\n[9] Saved: model/nb_model.pkl + model/vectorizer.pkl")
print("    Ready for Flask app!")