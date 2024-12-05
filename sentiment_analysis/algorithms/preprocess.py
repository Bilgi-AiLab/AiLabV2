import emoji
import re
from contractions import fix

def preprocess_text(text):
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Expand contractions
    text = fix(text)
    return text