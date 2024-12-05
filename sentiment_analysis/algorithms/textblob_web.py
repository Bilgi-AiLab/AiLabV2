from textblob import TextBlob
from sentiment_analysis.algorithms.preprocess import preprocess_text

def textblob(corpus):
    # Initialize counters and accumulators
    negative_doc_count = 0
    positive_doc_count = 0
    neutral_doc_count = 0
    polarity_score = 0

    # Store detailed scores for each document
    detailed_scores = []

    # Process each tweet in the corpus
    for i, tweet in enumerate(corpus):
        cleaned_tweet = preprocess_text(tweet)
        # Get sentiment from TextBlob
        blob = TextBlob(cleaned_tweet)
        polarity = blob.sentiment.polarity  # Get polarity score (-1 to 1)
        
        # Accumulate polarity score
        polarity_score += polarity
        
        # Categorize tweet based on polarity score
        sentiment = "Neutral"
        if polarity > 0.05:
            positive_doc_count += 1
            sentiment = "Positive"
        elif polarity < -0.05:
            negative_doc_count += 1
            sentiment = "Negative"
        else:
            neutral_doc_count += 1

        # Add detailed scores
        detailed_scores.append({
            "id": i + 1,
            "text": tweet,
            "compound": round(polarity,4),
            "positive": round(max(0, polarity), 4),
            "neutral": round(1 - abs(polarity),4),
            "negative": round(max(0, -polarity),4),
            "sentiment": sentiment
        })
    
    # Calculate average polarity score
    average_polarity_score = polarity_score / len(corpus) if len(corpus) > 0 else 0

    output = {
        "filecount": len(corpus),
        "polarity_value": float(average_polarity_score),
        "negative_doc_count": negative_doc_count,
        "positive_doc_count": positive_doc_count,
        "neutral_doc_count": neutral_doc_count,
        "detailed_scores": detailed_scores
    }

    return output
