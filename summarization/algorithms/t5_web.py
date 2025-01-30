from transformers import T5ForConditionalGeneration, T5Tokenizer
from summarization.algorithms.preprocessing import preprocess_text
from bert_score import score
import os

def t5(text, num_beams=5):
    
    #os.environ["WANDB_DISABLED"] = "true"
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    model_name="google-t5/t5-base"
    device = "cuda"
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