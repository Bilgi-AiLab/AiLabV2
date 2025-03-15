from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentiment_analysis.algorithms.preprocess_absa import preprocess_text
import re

def absaberturk(corpus):
    
    model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_berturk")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tokenizer_berturk")
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer) #0.93 accuracy for validation set, 0.85 accuracy for test set, 0.21 validation loss, 0.2 training loss, trained with TREMO dataset, 2 epoch, 2e-5 learning rate
    
    happiness_doc_count = 0
    sadness_doc_count = 0
    fear_doc_count = 0
    anger_doc_count = 0
    disgust_doc_count = 0
    surprise_doc_count = 0
    polarity_score = 0

    detailed_scores = []
    
    #corpus = [s.strip() for s in re.split(r'\s*&\s*', corpus) if s.strip()]
    
    for i, text in enumerate(corpus):
        cleaned_text = preprocess_text(text)
        result = sentiment_analyzer(cleaned_text)[0] 
        label = result['label'] 
        score = result['score'] 
        polarity = 0
        sentiment = "Unknown" 
       
        if label == "LABEL_0":
            polarity = score
            sentiment = "Happy"
            happiness_doc_count += 1
        elif label == "LABEL_3":
            polarity = score
            sentiment = "Sadness"
            sadness_doc_count += 1
        elif label == "LABEL_1":
            polarity = score
            sentiment = "Fear"
            fear_doc_count += 1
        elif label == "LABEL_2":
            polarity = score
            sentiment = "Anger"
            anger_doc_count += 1
        elif label == "LABEL_4":
            polarity = score
            sentiment = "Disgust"
            disgust_doc_count += 1
        elif label == "LABEL_5":
            polarity = score
            sentiment = "Surprised"
            surprise_doc_count += 1

        polarity_score += polarity

       
        detailed_scores.append({
            "id": i + 1,
            "text": text,
            "compound": round(polarity, 4),
            "sentiment": sentiment
        })


    average_polarity_score = polarity_score / len(corpus) if len(corpus) > 0 else 0

    output = {
        "filecount": len(corpus),
        "polarity_value": round(average_polarity_score, 4),
        "happiness_doc_count": happiness_doc_count,
        "sadness_doc_count": sadness_doc_count,
        "fear_doc_count": fear_doc_count,
        "anger_doc_count": anger_doc_count,
        "disgust_doc_count": disgust_doc_count,
        "surprise_doc_count": surprise_doc_count,
        "detailed_scores": detailed_scores
    }

    return output