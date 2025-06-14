from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from summarization.algorithms.preprocessing import preprocess_text
from rouge_score import rouge_scorer
from bert_score import score

def lexrank(tweets, num_sentences=5):
    combined_text = " ".join(tweets)

    # Preprocess the combined text
    combined_text = preprocess_text(combined_text)

    # Parse the combined text
    parser = PlaintextParser.from_string(combined_text, Tokenizer("english"))
    
    # Initialize LexRank summarizer
    summarizer = LexRankSummarizer()
    
    # Generate summary (take the top N sentences)
    summary_sentences = summarizer(parser.document, num_sentences)
    
    # Combine the sentences into a summary
    summary = " ".join(str(sentence) for sentence in summary_sentences)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(combined_text, summary)

    rouge1 = rouge_scores["rouge1"]
    rouge2 = rouge_scores["rouge2"]
    rougeL = rouge_scores["rougeL"]

    P, R, F1 = score([summary], [combined_text], lang="en", model_type="roberta-large")
    bertscore_f1 = F1.mean().item()

    output = {
        "summary": summary,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bert_score": bertscore_f1
    }
    
    return output
