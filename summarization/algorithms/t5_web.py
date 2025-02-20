from transformers import T5ForConditionalGeneration, T5Tokenizer
from summarization.algorithms.preprocessing import preprocess_text
from bert_score import score
import torch
import os

'''
def t5(text, num_beams=5):
    
    #os.environ["WANDB_DISABLED"] = "true"
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    model_name="google-t5/t5-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    texts =[ preprocess_text(txt) for txt in text]
    # Tokenize the input text
    inputs = tokenizer(texts, max_length=1024, truncation=True, padding=True, return_tensors="pt").to(device)
    
    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=60, 
        min_length=10, 
        length_penalty=2.0, 
        num_beams=num_beams, 
        early_stopping=True
    )
    
    # Decode the generated tokens to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    P, R, F1 = score([summary], [text], lang="en", model_type="roberta-large")
    bertscore_f1 = F1.mean().item()

    output = {
        "summary": summary,
        "bert_score": bertscore_f1
    }

    return output
'''

def t5(text, num_beams=5):
    # Initialize model and tokenizer
    model_name="google-t5/t5-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    processed_tweets = [preprocess_text(txt) for txt in text]

    def create_dynamic_chunks(tweets, max_tokens=1024):
        chunks = []
        current_chunk = []
        current_length = 0

        for tweet in tweets:
            tweet_length = len(tokenizer.encode(tweet, add_special_tokens=False))

            if current_length + tweet_length > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [tweet]
                current_length = tweet_length
            else:
                current_chunk.append(tweet)
                current_length += tweet_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks


    chunks = create_dynamic_chunks(processed_tweets)

    chunk_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt").to(device)

        chunk_length = len(inputs["input_ids"][0])
        max_summary_len = min(100, int(chunk_length * 0.2))  # 20% of chunk length

        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_summary_len,
            min_length=10,
            length_penalty=1.5,
            num_beams=num_beams,
            early_stopping=True
        )

        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(chunk_summary)

    
    combined_text = " ".join(chunk_summaries)
    inputs = tokenizer(combined_text, max_length=1024, truncation=True, return_tensors="pt").to(device)

    final_summary_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=20,
        length_penalty=1.5,
        num_beams=num_beams,
        early_stopping=True
    )

    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)

    original_text = " ".join(processed_tweets)
    P, R, F1 = score([final_summary], [original_text], lang="en", model_type="roberta-large")
    bertscore_f1 = F1.mean().item()

    output = {
        "summary": final_summary,
        "bert_score": bertscore_f1
    }

    torch.cuda.empty_cache()
    return output