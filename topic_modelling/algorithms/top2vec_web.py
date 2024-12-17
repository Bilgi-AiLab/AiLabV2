from top2vec import Top2Vec
from gensim.models import CoherenceModel
from topic_modelling.algorithms import preprocess_top2vec
import numpy as np

def top2vec(corpus):
    cleaned_data, data_tokens, id2word, corpus = preprocess_top2vec.preprocess(corpus=corpus)
    if not cleaned_data:
        raise ValueError("The cleaned data is empty. Please check the preprocessing steps.")
    
    doc_ids = [i for i in range(len(cleaned_data))]  
    
    print("Fitting Top2Vec model...")
    model = Top2Vec(documents=cleaned_data, embedding_model='all-MiniLM-L6-v2', speed='deep-learn', workers=4, document_ids=doc_ids, contextual_top2vec=True, min_count=5)
    
    topics_words, word_scores, topic_nums = model.get_topics()
    topic_sizes, topic_nums = model.get_topic_sizes()

    word_distributions = []
    for words, scores in zip(topics_words, word_scores):
        word_dist = []
        for word, score in zip(words, scores):
            word_dist.append((word,score))
        word_distributions.append(word_dist)

    doc_topic=model.get_document_topic_relevance()
    
    topic_distributions = []
    for document_id, scores in enumerate(doc_topic):
        topic_distribution = [
            [topic_idx, score] for topic_idx, score in enumerate(scores)
        ]
        topic_distributions.append(topic_distribution)
    
    valid_topics = {int(topic) for topic in set(topic_nums)}
    doc_dist = {topic: [] for topic in set(valid_topics)}

    for doc_id, relevance_scores in enumerate(doc_topic):
        most_relevant_topic = np.argmax(relevance_scores)
        doc_dist[most_relevant_topic].append(doc_ids[doc_id]) 
    
    
    top2vec_topics = [[word[0] for word in topic] for topic in word_distributions]
    coherence_model = CoherenceModel(topics=top2vec_topics, texts=data_tokens, dictionary=id2word, coherence='c_v').get_coherence()
    
    output = {
        "filecount":  len(data_tokens),
        "coherence_value": float(coherence_model),
        "word_distributions": word_distributions,
        "topic_distributions": topic_distributions, 
        "doc_dist": doc_dist, 
        "data_tokens": data_tokens,
    }
    
    return output
