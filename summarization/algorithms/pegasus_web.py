from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from summarization.algorithms.preprocessing import preprocess_text
from summac.model_summac import SummaCZS

def pegasus(text, num_beams=5):
    """
    Summarize text using the PEGASUS model.

    Args:
        text (str): The input text to summarize.
        model_name (str): The pre-trained PEGASUS model to use.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
        num_beams (int): Number of beams for beam search (higher gives better quality but slower).

    Returns:
        str: Generated summary.
    """
    # Load the tokenizer and model
    model_name="google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    texts =[ preprocess_text(txt) for txt in text]
    # Tokenize the input text
    inputs = tokenizer(texts, max_length=1024, truncation=True, return_tensors="pt")
    
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

    summac_model = SummaCZS(granularity="sentence", model_name="vitc") 
    evaluation_result = summac_model.score([{"text": text, "summary": summary}])
    summac_score = evaluation_result["scores"][0]

    output = {
        "summary": summary,
        "summac_score": summac_score
    }

    return output