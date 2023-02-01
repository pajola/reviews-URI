import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import opinion_lexicon as ol
import gensim
import gensim.downloader
import gensim.corpora as corpora
import re
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count, Pool


import warnings

#get english stopwords
sw = set(stopwords.words('english'))

#define the lemmatizer
lemmatizer = WordNetLemmatizer()

embedding_features = [
    "Embedding.LDA", "Embedding.GLOVE", "Embedding.Word2Vec" #100 + 300 + 300
]

def has_embedding(df):
    if len([c for c in df.columns if embedding_features[0] in c]) != 100:
        return False

    if len([c for c in df.columns if embedding_features[1] in c]) != 300:
        return False

    if len([c for c in df.columns if embedding_features[2] in c]) != 300:
        return False

    return True

def custom_tokenizer(x, remove_stop_words = False):
    #tokenize the sentence
    x_tok = word_tokenize(x)

    #lemmatize
    x_lemm = [lemmatizer.lemmatize(x) for x in x_tok]

    if remove_stop_words:
        x_lemm = [x for x in x_tok if x not in sw]

    return x_lemm

def lda_preprocess_sentence(i, x):

    #lowercase
    try:
        x = x.lower()
    except:
        raise Exception(i, x)

    #remove punctuation
    sentence = re.sub('[,\.!?]', '', x)

    return sentence

def lda_preprocessing(corpus):
    #preprocess the sentences
    corpus_preprocessed = [lda_preprocess_sentence(i, x) for i, x in enumerate(corpus)]

    #tokenize and remove stopwords
    corpus_tokenized = [custom_tokenizer(x, True) for x in corpus_preprocessed]

    return corpus_tokenized

def get_LDA(overall_corpus = None, train_corpus = None, val_corpus = None, test_corpus = None):
    #following towardsdatascience resource
    # https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

    #set the number of topics
    NTOPICS = 100

    #define the number of available cores
    pool = Pool(cpu_count() - 1)

    if overall_corpus is not None:
        #preprocess the corpus
        overall_corpus = lda_preprocessing(overall_corpus)
        # overall_corpus = pool.map(lda_preprocessing, overall_corpus)

        # Create Dictionary
        id2word = corpora.Dictionary(overall_corpus)#.filter_extremes(no_below = 5, keep_n = 10000)# create a vocabulary# create a vocabulary
        overall_corpus_gensim = [id2word.doc2bow(text) for text in overall_corpus] #transform

        #build lda topic
        lda_model = gensim.models.LdaMulticore(corpus = overall_corpus_gensim,
            id2word = id2word,
            num_topics = NTOPICS,
            workers = None #all available workers
        )

        #get the topic distribution
        all_topics_overall = lda_model.get_document_topics(overall_corpus_gensim)
        all_topics_csr = gensim.matutils.corpus2csc(all_topics_overall)
        overall_lda = all_topics_csr.T.toarray()

        return overall_lda

    else:
        #preprocess the corpus
        # train_corpus = lda_preprocessing(train_corpus)
        # val_corpus = lda_preprocessing(val_corpus)
        # test_corpus = lda_preprocessing(test_corpus)
        train_corpus = pool.map(lda_preprocessing, train_corpus)
        val_corpus = pool.map(lda_preprocessing, val_corpus)
        test_corpus = pool.map(lda_preprocessing, test_corpus)

        # Create Dictionary
        id2word = corpora.Dictionary(train_corpus)#.filter_extremes(no_below = 5, keep_n = 10000)# create a vocabulary

        train_corpus_gensim = [id2word.doc2bow(text) for text in train_corpus] #transform
        val_corpus_gensim = [id2word.doc2bow(text) for text in val_corpus] #transform
        test_corpus_gensim = [id2word.doc2bow(text) for text in test_corpus] #transform

        #build lda topic
        lda_model = gensim.models.LdaMulticore(corpus = train_corpus_gensim,
            id2word = id2word,
            num_topics = NTOPICS,
            workers = None #all available workers
        )

        # Compute Perplexity
        # print('\n\t\t\t--->Perplexity: ', lda_model.log_perplexity(train_corpus))

        #get the topic distribution
        all_topcis_train = lda_model.get_document_topics(train_corpus_gensim)
        all_topics_csr = gensim.matutils.corpus2csc(all_topcis_train)
        train_lda = all_topics_csr.T.toarray()

        all_topcis_val = lda_model.get_document_topics(val_corpus_gensim)
        all_topics_csr = gensim.matutils.corpus2csc(all_topcis_val)
        val_lda = all_topics_csr.T.toarray()

        all_topcis_test = lda_model.get_document_topics(test_corpus_gensim)
        all_topics_csr = gensim.matutils.corpus2csc(all_topcis_test)
        test_lda = all_topics_csr.T.toarray()

        return train_lda, val_lda, test_lda

def sentence_embedding(engine, sentence):
    #tokenize
    tokens = nltk.word_tokenize(sentence.lower())
    # print(f"Sentence: {len(tokens)}")
    cnt = 0
    embedding = np.zeros(300)

    #iterate over the words
    for t in tokens:
        #check if word is contained in the Dictionary
        if t in engine:
            embedding += np.array(engine[t])
            cnt += 1

    if cnt > 0:
        embedding /= cnt

    return embedding

def get_embeddings(corpus_overall = None, corpus_train = None, corpus_val = None, corpus_test = None, engine = None):
    #check the engine
    if engine not in list(gensim.downloader.info()['models'].keys()):
        raise Exception("Engine not found")
    else:
        #load the engine
        embedder = gensim.downloader.load(engine)

    #define the number of available cores
    # pool = Pool(cpu_count() - 1)

    if corpus_overall is not None:
        #define the output lists
        emb_overall = []
        #get the embeddings
        emb_overall = [sentence_embedding(embedder, sentence) for sentence in corpus_overall]


        return emb_overall
    else:
        #define the output lists
        emb_train, emb_val, emb_test = [], [], []

        #get the embeddings
        emb_train = [sentence_embedding(embedder, sentence) for sentence in corpus_train]
        emb_val = [sentence_embedding(embedder, sentence) for sentence in corpus_val]
        emb_test = [sentence_embedding(embedder, sentence) for sentence in corpus_test]

        return emb_train, emb_val, emb_test

def embedding_extractor(df):
    #extract the corpus
    corpus_overall = df['reviewText'].values

    if "Embedding.LDA.1#" not in df.columns:
        #get lda representation
        overall_lda = get_LDA(overall_corpus = corpus_overall)

        #convert to pandas
        c = [f"Embedding.LDA.{i}#" for i in range(0, 100)] #100 is the number of topics
        overall_lda_df = pd.DataFrame(overall_lda, columns = c)

        #concatenate the two dataframes
        df = pd.concat([df, overall_lda_df], axis = 1)


    if "Embedding.GLOVE.1#" not in df.columns:
        #get glove representation
        overall_glove = get_embeddings(corpus_overall = corpus_overall, engine = 'glove-wiki-gigaword-300')

        #convert to pandas
        c = [f"Embedding.GLOVE.{i}#" for i in range(0, 300)]
        overall_glove_df = pd.DataFrame(overall_glove, columns = c)

        #concatenate the two dataframes
        df = pd.concat([df, overall_glove_df], axis = 1)

    if "Embedding.Word2Vec.1#" not in df.columns:
        #get glove representation
        overall_w2v = get_embeddings(corpus_overall = corpus_overall, engine = 'word2vec-google-news-300')

        #convert to pandas
        c = [f"Embedding.Word2Vec.{i}#" for i in range(0, 300)]
        overall_w2v_df = pd.DataFrame(overall_w2v, columns = c)

        #concatenate the two dataframes
        df = pd.concat([df, overall_w2v_df], axis = 1)


    return df
