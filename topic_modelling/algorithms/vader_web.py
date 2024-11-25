from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

def vader(corpus):
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # Initialize counters and accumulators
    negative_doc_count = 0
    positive_doc_count = 0
    neutral_doc_count = 0
    polarity_score = 0

    # Process each tweet in the corpus
    for tweet in corpus:
        cleaned_tweet = preprocess_text(tweet)
        polarity_scores = analyzer.polarity_scores(cleaned_tweet)
        compound_score = polarity_scores['compound']
        
        # Accumulate polarity score
        polarity_score += compound_score
        
        # Categorize tweet based on compound score
        if compound_score < 0:
            negative_doc_count += 1
        elif compound_score > 0:
            positive_doc_count += 1
        else:
            neutral_doc_count += 1
    
    # Calculate average polarity score
    average_polarity_score = polarity_score / len(corpus) if len(corpus) > 0 else 0

    output = {
        "filecount": len(corpus),
        "polarity_value": float(average_polarity_score),
        "doc_count": {
            'Negative': negative_doc_count,
            'Positive': positive_doc_count,
            'Neutral': neutral_doc_count
        }
    }

    return output
