import nltk
import re
from nltk.tokenize import RegexpTokenizer
import gensim
from nltk.stem import WordNetLemmatizer
import os
from nltk.corpus import wordnet
import emoji
import contractions

def preprocess(corpus):
    file = open(f"{os.path.dirname(__file__)}/turkce-stop-words.txt")
    stops = [line.strip() for line in file.readlines()]
    replace_with_space = re.compile('[/(){}\[\]\|@,;]')
    remove_symbols1 = re.compile("[^0-9a-z_ğüşıöç .']")
    remove_urls = re.compile(r'http[s]?://\S+|www\.\S+')
    remove_mentions_and_hashtags = re.compile(r'[@#]\w+')
    stopwords = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('turkish')
    stopwords.extend(stops)
    remove_3chars = re.compile(r'\b\w{1,3}\b')

    def get_synonym(word):
        synonyms = wordnet.synsets(word)
        if synonyms:
            return synonyms[0].lemmas()[0].name()
        return word

    def clean_text(text):
        """
            text: a string

            return: modified initial string
        """
        valid_characters = 'abcçdefgğhıijklmnoöpqrsştuüvwxyzQWERTYUIOPĞÜASDFGHJKLŞİZXCVBNMÖÇ1234567890 '
        text = remove_urls.sub('', text)
        #text = remove_mentions_and_hashtags.sub('', text)
        text = contractions.fix(text)
        text = ''.join([x for x in text if x in valid_characters])
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        lower_map = {
            ord(u'I'): u'ı',
            ord(u'İ'): u'i',
            ord(u'Ö'): u'ö',
            ord(u'Ü'): u'ü',
            ord(u'Ş'): u'ş',
            ord(u'Ğ'): u'ğ',
        }
        text = text.translate(lower_map)
        text = text.lower()
        text = replace_with_space.sub(' ', text)
        text = remove_symbols1.sub('', text)
        text = remove_3chars.sub('', text)
        text = ' '.join([word for word in text.split() if word not in stopwords])
        text = ' '.join([get_synonym(word) for word in text.split()])
        text = emoji.replace_emoji(text, replace=' ')
        return text

    # First pass: Clean the text
    cleaned_data = [clean_text(news) for news in corpus]

    # Tokenization and frequency filtering
    tokenizer = RegexpTokenizer(r'\w+')
    data_tokens = [tokenizer.tokenize(doc) for doc in cleaned_data]

    # Create dictionary and calculate token frequency
    id2word = gensim.corpora.Dictionary(data_tokens)
    min_frequency = 5
    token_frequency = {word: count for word, count in id2word.dfs.items()}

    # Second pass: Filter tokens based on frequency
    filtered_tokens = [
        [word for word in doc if token_frequency.get(id2word.token2id.get(word, -1), 0) >= min_frequency]
        for doc in data_tokens
    ]

    # Create filtered corpus
    id2word.filter_tokens(bad_ids=[
        token_id for token_id, freq in id2word.dfs.items() if freq < min_frequency
    ])
    id2word.compactify()

    corpus = [id2word.doc2bow(doc) for doc in filtered_tokens]

    return cleaned_data, filtered_tokens, id2word, corpus
