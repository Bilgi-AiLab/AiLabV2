from transformers import BartForConditionalGeneration, BartTokenizer
from summarization.algorithms.preprocessing import preprocess_text
from rouge_score import rouge_scorer
import torch
from bert_score import score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

'''
def bart(text, num_beams=3):
    # Initialize model and tokenizer
    model_name="facebook/bart-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    processed_tweets = [preprocess_text(txt) for txt in text]

    def create_dynamic_chunks(tweets, max_tokens):
        chunks = []
        current_chunk = []
        current_length = 0

        for tweet in tweets:
            sentences = sent_tokenize(tweet)
            for sentence in sentences:
                sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))

                if current_length + sentence_length > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    #current_chunk.append(sentence + ". ")
                    current_chunk.append(sentence)
                    current_length += sentence_length

        if current_chunk:
            #chunks.append(" ".join(current_chunk) + ".")
            chunks.append(" ".join(current_chunk))
        return chunks


    chunks = create_dynamic_chunks(processed_tweets, 1024)

    chunk_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt").to(device)

        #chunk_length = len(inputs["input_ids"][0])
        #max_summary_len = min(100, int(chunk_length * 0.2))  # 20% of chunk length

        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=10,
            length_penalty=0.8,
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
        min_length=10,
        length_penalty=0.8,
        num_beams=num_beams,
        early_stopping=True
    )

    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)

    original_text = " ".join(processed_tweets)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(original_text, final_summary)

    rouge1 = rouge_scores["rouge1"]
    rouge2 = rouge_scores["rouge2"]
    rougeL = rouge_scores["rougeL"]

    P, R, F1 = score([final_summary], [original_text], lang="en", model_type="roberta-large")
    bertscore_f1 = F1.mean().item()

    output = {
        "summary": final_summary,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bert_score": bertscore_f1
    }

    torch.cuda.empty_cache()
    return output
'''
def bart(text, num_beams=3):
    model_name="facebook/bart-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    processed_tweets = [preprocess_text(txt) for txt in text]

    original_text = " ".join(processed_tweets)

    parser = PlaintextParser.from_string(original_text, Tokenizer("english"))

    summarizer = LexRankSummarizer()

    summary_sentences = summarizer(parser.document, 10)
    
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    inputs = tokenizer(summary, max_length=1024, truncation=True, return_tensors="pt").to(device)

    final_summary_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=10,
        length_penalty=0.8,
        num_beams=num_beams,
        early_stopping=True
    )

    final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)


    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(summary, final_summary)
    rouge1 = rouge_scores["rouge1"]
    rouge2 = rouge_scores["rouge2"]
    rougeL = rouge_scores["rougeL"]

    P, R, F1 = score([final_summary], [original_text], lang="en", model_type="roberta-large")
    bertscore_f1 = F1.mean().item()

    output = {
        "summary": final_summary,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bert_score": bertscore_f1
    }

    torch.cuda.empty_cache()
    return output