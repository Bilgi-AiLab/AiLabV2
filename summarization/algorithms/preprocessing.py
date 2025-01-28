import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
import os

# Download NLTK data (if not already done)
nltk.download("punkt")
nltk.download("stopwords")

def preprocess_text(text, remove_stopwords=False):
    
    # 1. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    remove_mentions_and_hashtags = re.compile(r'[@#]\w+')
    text = remove_mentions_and_hashtags.sub('', text)
    # 3. Remove special characters, emojis, and extra spaces
    text = re.sub(r"[^a-zA-Z0-9.,!?'\"]+", " ", text)  # Allow common punctuation
    text = re.sub(r"\s+", " ", text).strip()
    
    # 4. Lowercasing
    text = text.lower()
    
    # 6. Remove stop words (if enabled)
    if remove_stopwords:
        stopwords =  nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('turkish')
        file = open(f"{os.path.dirname(__file__)}/turkce-stop-words.txt")
        stops = [line.strip() for line in file.readlines()]
        stopwords.extend(stops)
        text = ' '.join([word for word in text.split() if word not in stopwords])
    
    return text