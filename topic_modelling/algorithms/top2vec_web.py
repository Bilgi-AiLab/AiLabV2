from top2vec import Top2Vec
from topic_modelling.algorithms import preprocess, coherence
import plotly.graph_objects as go


def top2vec_coherence(corpus, embedding_model='universal-sentence-encoder'):
    # Veri ön işleme
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    
    # Top2Vec modelini eğit
    model = Top2Vec(
        documents=cleaned_data,
        embedding_model=embedding_model,
        speed="fast-learn",
        workers=4  # Paralel işlem için
    )
    
    # Konuları ve anahtar kelimeleri al
    topic_words, word_scores, topic_nums, topic_scores = [], [], [], []
    num_topics = model.get_num_topics()

    for topic_num in range(num_topics):
        words, scores, docs, doc_scores = model.get_topic(topic_num)
        topic_words.append(words)
        word_scores.append(scores)
        topic_nums.append(topic_num)
        topic_scores.append(doc_scores.mean())  # Ortalama skor
    
    # Coherence Score Hesaplama
    coh_score = coherence.coherence_value(
        model=model, 
        tokens=data_tokens, 
        dictionary=id2word
    )
    
    return {
        "topic_words": topic_words,
        "word_scores": word_scores,
        "topic_nums": topic_nums,
        "coherence_score": coh_score,
        "topic_scores": topic_scores
    }


def Top2Vec_Model(corpus, embedding_model='universal-sentence-encoder'):
    # Veri ön işleme
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    
    # Modeli oluştur ve eğit
    model = Top2Vec(
        documents=cleaned_data,
        embedding_model=embedding_model,
        speed="fast-learn",
        workers=4
    )
    
    # Konular ve Anahtar Kelimeler
    topics, topic_scores, doc_topics, doc_scores = model.get_topics()
    
    # Konu Kelime Dağılımları
    word_distributions = {f"Topic {i+1}": topic for i, topic in enumerate(topics)}
    
    # Coherence Score
    coh_score = coherence.coherence_value(
        model=model,
        tokens=data_tokens,
        dictionary=id2word
    )
    
    # Çıktılar
    output = {
        "filecount": len(data_tokens),
        "coherence_value": coh_score,
        "topics": topics,
        "topic_scores": topic_scores,
        "doc_topics": doc_topics,
        "doc_scores": doc_scores,
        "data_tokens": data_tokens,
        "word_distributions": word_distributions  # Bu eklendi
    }
    print("DEBUG: Top2Vec_Model'den dönen output:")
    print(output)
    return output

    