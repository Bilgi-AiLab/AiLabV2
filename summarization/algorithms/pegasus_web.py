from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from summarization.algorithms.preprocessing import preprocess_text
from bert_score import score
import os
import gc
import torch

'''
def pegasus(text, num_beams=5):
    
    #os.environ["WANDB_DISABLED"] = "true"
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    model_name="google/pegasus-xsum"
    device = "cuda"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]

def pegasus(texts, num_beams=5, batch_size=8):
    # Preprocess texts
    processed_texts = [preprocess_text(txt) for txt in texts]

    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer in evaluation mode
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # Process text in batches
    summaries = []
    for text_batch in batch(processed_texts, batch_size):
        try:
            inputs = tokenizer(text_batch, max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
            torch.cuda.empty_cache()
            # Use mixed precision (float16) for better memory handling
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=60,
                    min_length=10,
                    length_penalty=2.0,
                    num_beams=num_beams,
                    early_stopping=True
                )

            summaries.extend([tokenizer.decode(s, skip_special_tokens=True) for s in summary_ids])

            # Clear GPU memory after each batch
            clear_memory()
        except torch.cuda.OutOfMemoryError:
            print("Out of memory in batch processing. Reducing batch size...")
            clear_memory()
            return {"summary": "Failed due to memory issues", "bert_score": 0.0}

    # Combine all batch summaries into a single summary
    combined_summary = " ".join(summaries)

    # Evaluate with BERTScore
    P, R, F1 = score([combined_summary], [" ".join(processed_texts)], lang="en", model_type="roberta-large")

    output = {
        "summary": combined_summary,
        "bert_score": F1.mean().item()
    }

    clear_memory()  # Final memory cleanup
    return output