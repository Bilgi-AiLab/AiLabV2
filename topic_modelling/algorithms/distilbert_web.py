from topic_modelling.algorithms import  preprocess, preprocess_bert
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from gensim.models import CoherenceModel
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances

def distilbert(corpus, n_topic):
    # Preprocess text using the given preprocess function
    cleaned_data, data_tokens, id2word, corpus = preprocess_bert.preprocess(corpus=corpus)
    if not cleaned_data:
        raise ValueError("The cleaned data is empty. Please check the preprocessing steps.")

    doc_number = len(cleaned_data)
    print(f"Documents: {doc_number}")

    # Load DistilBERT model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Generate embeddings for each document
    def get_embeddings(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        # Use the [CLS] token's embedding as the document representation
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    document_embeddings = get_embeddings(cleaned_data)

    # Reduce dimensionality for clustering
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(document_embeddings)

    # Cluster embeddings using KMeans
    kmeans = KMeans(n_clusters=n_topic, random_state=42)
    topics = kmeans.fit_predict(reduced_embeddings)

    # Calculate topic-word distributions
    word_distributions = []
    for topic_id in range(n_topic):
        topic_docs = [data_tokens[idx] for idx, topic in enumerate(topics) if topic == topic_id]
        if topic_docs:
            # Flatten all tokens in the topic and count word frequencies
            all_tokens = [token for doc in topic_docs for token in doc]
            word_freq = Counter(all_tokens)
            total_words = sum(word_freq.values())
            # Use id2word to map token IDs back to words
            top_words = word_freq.most_common(15)
            word_distributions.append([(id2word[word] if word in id2word else str(word), freq / total_words) for word, freq in top_words])
        else:
            word_distributions.append([])

    # Generate document-topic distributions
    distances = euclidean_distances(reduced_embeddings, kmeans.cluster_centers_)
    probs = 1 / (1 + distances)
    topic_distributions = []
    for document_idx, document_probs in enumerate(probs):
        topic_distribution = [
            [topic_idx, prob] for topic_idx, prob in enumerate(document_probs)
        ]
        topic_distributions.append(topic_distribution)

    # Assign documents to topics
    doc_dist = {topic: [] for topic in range(n_topic)}
    for doc_idx, topic in enumerate(topics):
        doc_dist[topic].append(doc_idx)

    # Calculate coherence value
    bert_topics = [[word[0] for word in topic if word] for topic in word_distributions]
    coherence_model = CoherenceModel(topics=bert_topics, texts=data_tokens, dictionary=id2word, coherence='c_v').get_coherence()

    output = {
        "filecount": doc_number,
        "coherence_value": float(coherence_model),
        "word_distributions": word_distributions,
        "topic_distributions": topic_distributions,
        "doc_dist": doc_dist,
        "data_tokens": data_tokens,
    }

    return output