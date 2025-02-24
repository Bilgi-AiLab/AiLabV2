import re
import nltk
import os

# Download NLTK data (if not already done)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

def preprocess_text(text):
    
    # 1. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove @mentions but keep hashtags
    text = re.sub(r"@\w+", "", text)  # Removes @user but keeps #hashtags

    # 4. Keep important symbols and common punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?'\-%$&/]+", " ", text)  

    # 5. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Ensure the sentence ends with a punctuation mark
    if text and not re.search(r"[.!?]$", text):
        text += "."
    
    
    return text