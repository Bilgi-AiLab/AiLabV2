from openai import OpenAI
from summarization.algorithms.preprocessing import preprocess_text
from rouge_score import rouge_scorer
from bert_score import score
import re

def deepseek(text):
    processed_tweets = [preprocess_text(txt) for txt in text]
    original_text = " ".join(processed_tweets)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(

    model="deepseek/deepseek-r1-distill-llama-70b:free",
    messages=[
        {
            "role": "system",
            "content": "Your task is to summarize the given text and only to return the generated summary to the user."
        },
        {
            "role": "user",
            "content": f"Summarize the following texts between 50 and 200 words: \n\n{original_text}"
        }
      ]
    )
    final_summary = completion.choices[0].message.content

    if "</think>" in final_summary:
        final_summary = re.sub(r'.*</think>', '', final_summary, flags=re.DOTALL).strip()

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
    return output
