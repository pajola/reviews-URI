import pandas as pd
import numpy as np
import spacy
from nltk import word_tokenize, sent_tokenize

structural_features = [
    "StructFeat.NTok", "StructFeat.NSent" , "StructFeat.NChar", "StructFeat.AvgSentLen",
    "StructFeat.AvgChar", "StructFeat.PercQuestSent", "StructFeat.NumQuestSent",
    "StructFeat.NumExclChars",  "StructFeat.NumExclSent",
    "StructFeat.LEX_ADJ_CNT", "StructFeat.LEX_ADP_CNT", "StructFeat.LEX_ADV_CNT",
    "StructFeat.LEX_AUX_CNT", "StructFeat.LEX_CONJ_CNT", "StructFeat.LEX_DET_CNT", "StructFeat.LEX_INTJ_CNT",
    "StructFeat.LEX_NOUN_CNT", "StructFeat.LEX_PART_CNT", "StructFeat.LEX_PRON_CNT", "StructFeat.LEX_PROPN_CNT",
    "StructFeat.LEX_PUNCT_CNT", "StructFeat.LEX_SCONJ_CNT", "StructFeat.LEX_SYM_CNT", "StructFeat.LEX_VERB_CNT",
    "StructFeat.LEX_X_CNT", "StructFeat.LEX_ADJ_RATIO", "StructFeat.LEX_ADP_RATIO",
"StructFeat.LEX_ADV_RATIO", "StructFeat.LEX_AUX_RATIO", "StructFeat.LEX_CONJ_RATIO",
"StructFeat.LEX_DET_RATIO", "StructFeat.LEX_INTJ_RATIO", "StructFeat.LEX_NOUN_RATIO", "StructFeat.LEX_PART_RATIO",
"StructFeat.LEX_PRON_RATIO", "StructFeat.LEX_PROPN_RATIO", "StructFeat.LEX_PUNCT_RATIO",
"StructFeat.LEX_SCONJ_RATIO", "StructFeat.LEX_SYM_RATIO", "StructFeat.LEX_VERB_RATIO",
"StructFeat.LEX_X_RATIO"
]

def has_structural(df):
    return len([c for c in df.columns if c in structural_features]) == len(structural_features)

def extract_lexical_features(text):
    nlp_engine = spacy.load("en_core_web_sm")

    #prepare the reuslts
    counter = {
        'ADJ' : 0, #adjective
        'ADP' : 0, #adpositions
        'ADV' : 0, #adverbs
        'AUX' : 0, #auxiliary verb
        'CONJ' : 0, #coordinating conjunction
        'DET' : 0, # determiner
        'INTJ' : 0, # interjection
        'NOUN' : 0, #noun
        'NUM' : 0, # numeral
        'PART' : 0, # particle
        'PRON' : 0, # pronoun
        'PROPN' : 0, # proper noun
        'PUNCT' : 0, # punctuation
        'SCONJ' : 0, # subordinating conjunction
        'SYM' : 0, # symbol
        'VERB' : 0, # verb
        'X' : 0 # other
    }

    #analyze the document
    doc = nlp_engine(text)

    for token in doc:
        if token.pos_ in counter.keys():
            counter[str(token.pos_)] += 1

    #get normalized count
    counter_ratio = counter.copy()

    if len(doc) != 0:
        for k in counter.keys():
            counter_ratio[k] /= len(doc)

    return counter, counter_ratio

def is_comparative_sentence(sentence):
    return False

def structural_extractor(df):
    #get the set of columns
    cols_set = set(df.columns)

    #extract the sentences
    reviews = df["reviewText"].tolist()

    #features that requires tokenizations
    F = set(["StructFeat.NTok", "StructFeat.NSent", "StructFeat.NChar", "StructFeat.AvgSentLen", "StructFeat.AvgChar"])
    if not F.issubset(cols_set):
        #tokenize
        reviews_tok = [word_tokenize(x) for x in reviews]
        reviews_sent = [sent_tokenize(x) for x in reviews]

        #review length
        if "StructFeat.NTok" not in df.columns:
            df["StructFeat.NTok"] = [len(x) for x in reviews_tok]

        # number of sentences: F2
        if "StructFeat.NSent" not in df.columns:
            df["StructFeat.NSent"] = [len(x) for x in reviews_sent]

        if "StructFeat.NChar" not in df.columns:
            df["StructFeat.NChar"] = df["reviewText"].apply(lambda x: len(x))

        #Average sentence length: F3
        if "StructFeat.AvgSentLen" not in df.columns:
            df["StructFeat.AvgSentLen"] = [len(x)/len(y) for x, y in zip(reviews_tok, reviews_sent)]

        if "StructFeat.AvgChar" not in df.columns:
            df["StructFeat.AvgChar"] = [np.sum([len(w)for w  in review])/len(review) for review in reviews_tok]

    #percentage of question sentences: F4
    if "StructFeat.PercQuestSent" not in df.columns:
        df["StructFeat.PercQuestSent"] = [(len([x for x in rs if "?" in x])/len(rs))*100 for rs in reviews_sent]
        df["StructFeat.NumQuestSent"] = [len([x for x in rs if "?" in x]) for rs in reviews_sent]

    #number of exclamatory marks: F5
    if "StructFeat.NumExclChars" not in df.columns:
        df["StructFeat.NumExclChars"] = [len([c for c in r if "!" == c]) for r in reviews]

    if "StructFeat.NumExclSent" not in df.columns:
        df["StructFeat.NumExclSent"] = [len([x for x in rs if "!" in x]) for rs in reviews_sent]

    #part-of-speech
    if "StructFeat.LEX_ADJ_CNT" not in df.columns:
        LEX = [extract_lexical_features(x) for x in df['reviewText'].tolist()]
        df["StructFeat.LEX_ADJ_CNT"] = [x[0]["ADJ"] for x in LEX]
        df["StructFeat.LEX_ADP_CNT"] = [x[0]["ADP"] for x in LEX]
        df["StructFeat.LEX_ADV_CNT"] = [x[0]["ADV"] for x in LEX]
        df["StructFeat.LEX_AUX_CNT"] = [x[0]["AUX"] for x in LEX]
        df["StructFeat.LEX_CONJ_CNT"] = [x[0]["CONJ"] for x in LEX]
        df["StructFeat.LEX_DET_CNT"] = [x[0]["DET"] for x in LEX]
        df["StructFeat.LEX_INTJ_CNT"] = [x[0]["INTJ"] for x in LEX]
        df["StructFeat.LEX_NOUN_CNT"] = [x[0]["NOUN"] for x in LEX]
        df["StructFeat.LEX_PART_CNT"] = [x[0]["PART"] for x in LEX]
        df["StructFeat.LEX_PRON_CNT"] = [x[0]["PRON"] for x in LEX]
        df["StructFeat.LEX_PROPN_CNT"] = [x[0]["PROPN"] for x in LEX]
        df["StructFeat.LEX_PUNCT_CNT"] = [x[0]["PUNCT"] for x in LEX]
        df["StructFeat.LEX_SCONJ_CNT"] = [x[0]["SCONJ"] for x in LEX]
        df["StructFeat.LEX_SYM_CNT"] = [x[0]["SYM"] for x in LEX]
        df["StructFeat.LEX_VERB_CNT"] = [x[0]["VERB"] for x in LEX]
        df["StructFeat.LEX_X_CNT"] = [x[0]["X"] for x in LEX]


        df["StructFeat.LEX_ADJ_RATIO"] = [x[1]["ADJ"] for x in LEX]
        df["StructFeat.LEX_ADP_RATIO"] = [x[1]["ADP"] for x in LEX]
        df["StructFeat.LEX_ADV_RATIO"] = [x[1]["ADV"] for x in LEX]
        df["StructFeat.LEX_AUX_RATIO"] = [x[1]["AUX"] for x in LEX]
        df["StructFeat.LEX_CONJ_RATIO"] = [x[1]["CONJ"] for x in LEX]
        df["StructFeat.LEX_DET_RATIO"] = [x[1]["DET"] for x in LEX]
        df["StructFeat.LEX_INTJ_RATIO"] = [x[1]["INTJ"] for x in LEX]
        df["StructFeat.LEX_NOUN_RATIO"] = [x[1]["NOUN"] for x in LEX]
        df["StructFeat.LEX_PART_RATIO"] = [x[1]["PART"] for x in LEX]
        df["StructFeat.LEX_PRON_RATIO"] = [x[1]["PRON"] for x in LEX]
        df["StructFeat.LEX_PROPN_RATIO"] = [x[1]["PROPN"] for x in LEX]
        df["StructFeat.LEX_PUNCT_RATIO"] = [x[1]["PUNCT"] for x in LEX]
        df["StructFeat.LEX_SCONJ_RATIO"] = [x[1]["SCONJ"] for x in LEX]
        df["StructFeat.LEX_SYM_RATIO"] = [x[1]["SYM"] for x in LEX]
        df["StructFeat.LEX_VERB_RATIO"] = [x[1]["VERB"] for x in LEX]
        df["StructFeat.LEX_X_RATIO"] = [x[1]["X"] for x in LEX]

    return df
