import re
from nltk.corpus import stopwords

def preprocess(text):
    """
    Preprocess email text by:
    1. Converting to lowercase
    2. Removing HTML tags
    3. Removing non-alphabetic characters
    4. Removing extra whitespace
    5. Removing English stop words
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in text.split() if t not in stop_words]
    return ' '.join(tokens)
