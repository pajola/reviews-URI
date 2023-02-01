import pandas as pd
import numpy as np
import textstat
from nltk import word_tokenize, sent_tokenize
from hunspell import Hunspell #spell checker

def isascii(s):
    """Check if the characters in string s are in ASCII, U+0-U+7F."""
    return len(s) == len(s.encode())

readability_features = [
    "Readability.NumMiss", "Readability.AVGMiss_Char", "Readability.NumMissExt",
    "Readability.NWords1C", "Readability.NWords29C", "Readability.NWords10C",
    "Readability.FKRE", "Readability.FKGL", "Readability.GFI", "Readability.SMOG",
    "Readability.ARI", "Readability.CLI"
]

def has_readability(df):
    return len([c for c in df.columns if c in readability_features]) == len(readability_features)

def load_requirements(df):
    #load the spell checker
    if "Readability.AVGMiss_Char" not in df.columns:
        load_spell_checker()

def load_spell_checker():
    #
    global speller_engine
    speller_engine = Hunspell()
    
    global speller_engine_extender
    speller_engine_extender = Hunspell()
    with open("./Sources/enwiki-latest-all-titles-in-ns0", "r") as file:
        words = file.readlines()
    #augment
    for w in words:
        if isascii(w):
            speller_engine_extender.add(w)

def readability_extractor(df):
    if "Readability.AVGMiss_Char" not in df.columns:
        #tokenize the words
        reviews_tok = [word_tokenize(x) for x in df["reviewText"].values]
        df["Readability.NumMiss"] = [np.sum([not speller_engine.spell(t) for t in s]) for s in reviews_tok]
        df["Readability.AVGMiss_Char"] = df["Readability.NumMiss"].values / df["StructFeat.NChar"].values
        df["Readability.NumMissExt"] = [np.sum([not speller_engine_extender.spell(t) for t in s]) for s in reviews_tok]

    #get the set of columns
    cols_set = set(df.columns)
    F = set(["Readability.NWords1C", "Readability.NWords29C", "Readability.NWords10C"])
    if not F.issubset(cols_set):
        if "review_tok" not in locals():
            reviews_tok = [word_tokenize(x) for x in df["reviewText"].values]

        #add the three columns
        df["Readability.NWords1C"] = [len([tok for tok in sentence if len(tok) == 1]) for sentence in reviews_tok]
        df["Readability.NWords29C"] = [len([tok for tok in sentence if len(tok) > 1 and len(tok) < 10]) for sentence in reviews_tok]
        df["Readability.NWords10C"] = [len([tok for tok in sentence if len(tok) >= 10]) for sentence in reviews_tok]

    if "Readability.FKRE" not in df.columns:
        df["Readability.FKRE"] = df['reviewText'].apply(lambda x: textstat.flesch_reading_ease(x))
        df["Readability.FKGL"] = df['reviewText'].apply(lambda x: textstat.flesch_kincaid_grade(x))
        df["Readability.GFI"] = df['reviewText'].apply(lambda x: textstat.gunning_fog(x))
        df["Readability.SMOG"] = df['reviewText'].apply(lambda x: textstat.smog_index(x))
        df["Readability.ARI"] = df['reviewText'].apply(lambda x: textstat.automated_readability_index(x))
        df["Readability.CLI"] = df['reviewText'].apply(lambda x: textstat.coleman_liau_index(x))

    return df
