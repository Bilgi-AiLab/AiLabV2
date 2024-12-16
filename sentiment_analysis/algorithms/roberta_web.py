from transformers import pipeline
from sentiment_analysis.algorithms.preprocess import preprocess_text

def roberta(corpus):
    # Initialize Hugging Face sentiment analysis pipeline with RoBERTa
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    # Initialize counters and accumulators
    negative_doc_count = 0
    positive_doc_count = 0
    neutral_doc_count = 0
    polarity_score = 0

    # Store detailed scores for each document
    detailed_scores = []

    # Process each document in the corpus
    for i, text in enumerate(corpus):
        cleaned_text = preprocess_text(text)  # Preprocess text
        result = sentiment_analyzer(cleaned_text)[0]  # Analyze sentiment
        label = result['label']  # Get sentiment label (e.g., Positive, Negative, Neutral)
        score = result['score']  # Get confidence score

        # Convert labels to numerical polarity for consistency
        if label == "LABEL_2":  # Positive
            polarity = score
            sentiment = "Positive"
            positive_doc_count += 1
        elif label == "LABEL_0":  # Negative
            polarity = -score
            sentiment = "Negative"
            negative_doc_count += 1
        else:  # LABEL_1 for Neutral
            polarity = 0
            sentiment = "Neutral"
            neutral_doc_count += 1

        polarity_score += polarity

        # Add detailed scores
        detailed_scores.append({
            "id": i + 1,
            "text": text,
            "compound": round(polarity, 4),
            "positive": round(max(0, polarity), 4),
            "neutral": round(1 - abs(polarity), 4),
            "negative": round(max(0, -polarity), 4),
            "sentiment": sentiment
        })

    # Calculate average polarity score
    average_polarity_score = polarity_score / len(corpus) if len(corpus) > 0 else 0

    output = {
        "filecount": len(corpus),
        "polarity_value": round(average_polarity_score, 4),
        "negative_doc_count": negative_doc_count,
        "positive_doc_count": positive_doc_count,
        "neutral_doc_count": neutral_doc_count,
        "detailed_scores": detailed_scores
    }

    return output
