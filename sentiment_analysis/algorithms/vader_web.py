from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentiment_analysis.algorithms.preprocess import preprocess_text

def vader(corpus):
    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
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
        polarity_scores = analyzer.polarity_scores(cleaned_tweet)
        compound_score = polarity_scores['compound']
        
        # Accumulate polarity score
        polarity_score += compound_score
        
        # Categorize tweet based on compound score
        sentiment = "Neutral"
        if compound_score > 0.05:
            positive_doc_count += 1
            sentiment = "Positive"
        elif compound_score < -0.05:
            negative_doc_count += 1
            sentiment = "Negative"
        else:
            neutral_doc_count += 1

        # Add detailed scores
        detailed_scores.append({
            "id": i + 1,
            "text": tweet,
            "compound": compound_score,
            "positive": polarity_scores['pos'],
            "neutral": polarity_scores['neu'],
            "negative": polarity_scores['neg'],
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
