from bertopic import BERTopic
from gensim.models import CoherenceModel
import plotly.graph_objects as go
from umap import UMAP
from topic_modelling.algorithms import distributions, preprocess_bert, preprocess, coherence, topic_distance
from gensim.corpora import Dictionary
import re
from sentence_transformers import SentenceTransformer


'''
-When you need to capture nuanced, contextual relationships in text data.
-When working with short, informal text like social media posts, customer reviews, or news headlines.
-When you don’t want to predefine the number of topics but need a flexible, 
data-driven approach to identify themes in your data.
'''

def bertopic(corpus, n_topic, is_turkish="False"):
    cleaned_data, data_tokens, id2word, corpus = preprocess_bert.preprocess(corpus=corpus, is_turkish=is_turkish)
    if not cleaned_data:
        raise ValueError("The cleaned data is empty. Please check the preprocessing steps.")

    doc_number = len(data_tokens)
    print(f"Documents: {doc_number}")

    umap_model = UMAP(n_neighbors=15, min_dist=0.05, n_components=5, random_state=42)
    
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(cleaned_data, batch_size=64, show_progress_bar=True)

    topic_model = BERTopic(calculate_probabilities=True, min_topic_size=5, umap_model=umap_model, nr_topics=n_topic, top_n_words=15, zeroshot_min_similarity=0.85, embedding_model=None, low_memory=True)
    
    topics, probs = topic_model.fit_transform(cleaned_data, embeddings=embeddings)
    
    #coherence_model = coherence.coherence_value(model=topic_model, tokens=data_tokens, dictionary=id2word)
    valid_topics = {topic for topic in set(topics) if topic != -1}
    
    topics_list = topic_model.get_topics()
    if -1 in topics_list:
        del topics_list[-1]
    
    word_distributions = []
    for topic_id, topic_words in topics_list.items():
        filtered_words = [(word, score) for word, score in topic_words if word and word.strip()]
        if filtered_words:
            word_distributions.append(filtered_words)
    '''
    topic_distributions = []
    
    topic_dist = []
    for topic_id in valid_topics:
        topic_dist.append([topic_id, probs[topic_id]])
    topic_distributions.append(topic_dist)
    '''
    topic_distributions = []
    
    for document_idx, document_probs in enumerate(probs):
        topic_distribution = [
            [topic_idx, prob] for topic_idx, prob in enumerate(document_probs) if prob > 0
        ]
        topic_distributions.append(topic_distribution)
    
    doc_dist = {topic: [] for topic in valid_topics}
    for doc_idx, topic in enumerate(topics):
        if topic != -1:
            doc_dist[topic].append(doc_idx)
    '''
    tmp, _ = topic_model.transform(cleaned_data)

    distributions = [[[topic, prob]] for topic, prob in zip(tmp, _)]
    doc_topic_distributions = [topic for topic in distributions if topic[0][0] != -1]
    
    doc_dist = {topic: [] for topic in set(topics) if topic != -1}  
    for doc_idx, topic in enumerate(topics):
        if topic != -1: 
            doc_dist[topic].append(doc_idx)
    '''
    
    bert_topics = [[word[0] for word in topic] for topic in word_distributions]
    
    try:
        coherence_model = CoherenceModel(topics=bert_topics, texts=data_tokens, dictionary=id2word, coherence='c_v').get_coherence()
    except Exception:
        coherence_model = 0
    
    output = {
        "filecount": doc_number,
        "coherence_value": float(coherence_model),
        "word_distributions": word_distributions,
        "topic_distributions": topic_distributions, 
        "doc_dist": doc_dist, 
        "data_tokens": data_tokens,
    }
    
    return output