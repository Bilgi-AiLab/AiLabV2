from django.test import TestCase

import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
# Step 1: Read the Excel file
# Replace 'your_file.xlsx' with the path to your Excel file
df = pd.read_excel('./TurkishTweets.xlsx')
df = df.dropna()
# Step 2: Extract the "Tweet" and "Etiket" columns
tweets = df['Tweet'].tolist()  
labels = df['Etiket'].tolist()

tweets = [str(tweet) for tweet in df['Tweet'].tolist()]
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_berturk")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tokenizer_berturk")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
acc = 0
count = 0
for text, label in zip(tweets, labels):
    tmp = ""
    result = sentiment_analyzer(text)[0]
    if label == "kızgın":
        tmp = "LABEL_2"  
    elif label == "korku":
        tmp = "LABEL_1"
    elif label == "mutlu":
        tmp = "LABEL_0"
    elif label == "surpriz":
        tmp = "LABEL_5"
    elif label == "üzgün":
        tmp = "LABEL_3"

    if tmp == result['label']:
        acc += 1
    
    if count % 1000 == 0:
        print(f"Epoch: {count}")
    count += 1

print(f"Accuracy: %{(acc / len(tweets)) * 100}")