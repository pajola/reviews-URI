import pandas as pd
import numpy as np

import re
import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import opinion_lexicon as ol

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer



#get english stopwords
sw = set(stopwords.words('english'))

#define the stemmer
stemmer_snowball = SnowballStemmer('english')


emotions_features = [
    "Emotions.Vader_POS", "Emotions.Vader_NEG", "Emotions.Vader_NEU",
    "Emotions.SentiWordNet_POS", "Emotions.SentiWordNet_OBJ", "Emotions.SentiWordNet_NEG",
    "Emotions.OpinionLexicon_POS", "Emotions.OpinionLexicon_NEG",
    'Emotions.GALC_unk', 'Emotions.GALC_admiration', 'Emotions.GALC_amusement',
    'Emotions.GALC_anger', 'Emotions.GALC_anxiety', 'Emotions.GALC_beingtouched',
    'Emotions.GALC_boredom', 'Emotions.GALC_compassion', 'Emotions.GALC_contempt',
    'Emotions.GALC_contentment', 'Emotions.GALC_desperation', 'Emotions.GALC_disappointment',
    'Emotions.GALC_disgust', 'Emotions.GALC_dissatisfaction', 'Emotions.GALC_envy',
    'Emotions.GALC_fear', 'Emotions.GALC_feelinglove', 'Emotions.GALC_gratitude',
    'Emotions.GALC_guilt', 'Emotions.GALC_happiness', 'Emotions.GALC_hatred',
    'Emotions.GALC_hope', 'Emotions.GALC_humilty', 'Emotions.GALC_interest',
    'Emotions.GALC_irritation', 'Emotions.GALC_jealousy', 'Emotions.GALC_joy',
    'Emotions.GALC_longing', 'Emotions.GALC_lust', 'Emotions.GALC_pleasure',
    'Emotions.GALC_pride', 'Emotions.GALC_relaxation', 'Emotions.GALC_relief',
    'Emotions.GALC_sadness', 'Emotions.GALC_shame', 'Emotions.GALC_surprise',
    'Emotions.GALC_tension', 'Emotions.GALC_positive', 'Emotions.GALC_negative',
]

def has_emotions(df):
    return len([c for c in df.columns if c in emotions_features]) == len(emotions_features)

def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def sentiwordnet(x):
    #CREDITS: https://nlpforhackers.io/sentiment-analysis-intro/

    #define the lemmatizer
    lemmatizer = WordNetLemmatizer()

    ## preprocessing
    x = x.lower() #lowercase
    x = re.sub(r'\B#\S+','',x)#remove hashtag
    x = re.sub(r"http\S+", "", x) #remove links
    x = ' '.join(re.findall(r'\w+', x)) #remove the Special characters from the text
    x = re.sub(r'\s+', ' ', x, flags=re.I) #Code to substitute the multiple spaces with single spaces
    x = re.sub(r'\s+[a-zA-Z]\s+', '', x) #Code to remove all the single characters in the text

    #remove stopwords
    x = word_tokenize(x)
    x = [w for w in x if w not in sw]

    #pos tag
    x_tag = nltk.pos_tag(x)

    #define the three scores
    pos, obj, neg, cnt = 0, 0, 0, 0
    for word, tag in x_tag:
        wn_tag = penn_to_wn(tag)

        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        #lemmatize
        lemma = lemmatizer.lemmatize(word, pos = wn_tag)
        if not lemma:
            continue

        #synsets
        synsets = wn.synsets(lemma, pos = wn_tag)
        if not synsets:
            continue

        #take the most common sense
        synset = synsets[0]
        # raise Exception("DEB", synset)
        sent = swn.senti_synset(synset.name())
        pos += sent.pos_score()
        obj += sent.obj_score()
        neg += sent.neg_score()

        cnt+=1
    #average
    if cnt > 0:
        pos /= cnt
        obj /= cnt
        neg /= cnt

    return pos, obj, neg

def opinion_lexicon(x):
    #get both positive and negative words
    positive_set = ol.positive()
    negative_set = ol.negative()

    #define the lemmatizer
    lemmatizer = WordNetLemmatizer()

    ## preprocessing
    x = x.lower() #lowercase
    x = re.sub(r'\B#\S+','',x)#remove hashtag
    x = re.sub(r"http\S+", "", x) #remove links
    x = ' '.join(re.findall(r'\w+', x)) #remove the Special characters from the text
    x = re.sub(r'\s+', ' ', x, flags=re.I) #Code to substitute the multiple spaces with single spaces
    x = re.sub(r'\s+[a-zA-Z]\s+', '', x) #Code to remove all the single characters in the text

    #remove stopwords
    x = nltk.word_tokenize(x)
    x = [w for w in x if w not in sw]

    #pos tag
    x_tag = nltk.pos_tag(x)

    #define the three scores
    pos, obj, neg, cnt = 0, 0, 0, 0
    for word, tag in x_tag:
        wn_tag = penn_to_wn(tag)

        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        #lemmatize
        lemma = lemmatizer.lemmatize(word, pos = wn_tag)
        if not lemma:
            continue

        #synsets
        synsets = wn.synsets(lemma, pos = wn_tag)
        if not synsets:
            continue

        #take the most common sense
        synset = synsets[0].lemma_names()[0]

        # print(synset)
        if synset in positive_set:
            pos += 1

        if synset in negative_set:
            neg += 1

        cnt+=1

    #average
    if cnt > 0:
        pos /= cnt
        neg /= cnt

    return pos, neg


galc_lexicon = {
    "admiration": {"admir*", "ador*", "awe*", "dazed", "dazzl*", "enrapt*", "enthrall*", "fascina*", "marveli*", "rapt*", "reveren*", "spellbound", "wonder*", "worship*"},
    "amusement": {"amus*",	"fun*", "humor*", "laugh*", "play*", "rollick*", "smil*"},
    "anger": {"anger", "angr*", "cross*" , "enrag*", "furious", "fury", "incens*", "infuriat*", "irate", "ire*", "mad*", "rag*", "resent*",	"temper",	"wrath*",	"wrought*"},
    "anxiety": {"anguish*", "anxi*", "apprehens*", "diffiden*", "jitter*", "nervous*", "trepida*", "wari*", "wary", "worried*", "worry*"},
    "beingtouched": {"affect*", "mov*", "touch*"},
    "boredom": {"bor*", "ennui", "indifferen*", "languor*", "tedi*", "wear*"},
    "compassion": {"commiser*", "compass*", "empath*", "pit*"},
    "contempt": {"contempt*", "denigr*", "deprec*", "deris*", "despi*", "disdain*", "scorn*"},
    "contentment": {"comfortabl*", "content*", "satisf*"},
    "desperation": {"deject*", "desolat*", "despair*", "desperat*", "despond*", "disconsolat*", "hopeless*", "inconsol*"},
    "disappointment": {"comedown	disappoint*", "discontent*", "disenchant*", "disgruntl*", "disillusion*", "frustrat*", "jilt*", "letdown", "resign*", "sour*", "thwart*"},
    "disgust": {"abhor*", "avers*", "detest*", "disgust*", "dislik*", "disrelish", "distast*", "loath*", "nause*", "queas*", "repugn*", "repuls*", "revolt*", "sicken*"},
    "dissatisfaction": {"dissatisf*", "unhapp*"},
    "envy": {"envious*", "envy*"},
    "fear": {"afraid*", "aghast*", "alarm*", "dread*", "fear*", "fright*", "horr*", "panic*", "scare*", "terror*"},
    "feelinglove": {"affection*", "fond*", "love*", "friend*","tender*"},
    "gratitude": {"grat*","thank*"},
    "guilt": {"blame*", "contriti*", "guilt*", "remorse*", "repent*"},
    "happiness": {"cheer*", "bliss*", "delect*", "delight*", "enchant*", "enjoy*", "felicit*", "happ*", "merr*"},
    "hatred": {"acrimon*", "hat*", "rancor*"},
    "hope": {"buoyan*", "confident*", "faith*", "hop*", "optim*"},
    "humilty": {"devout*", "humility"},
    "interest": {"absor*", "alert", "animat*", "ardor*", "attenti*", "curi*", "eager*", "enrapt*", "engross*", "enthusias*", "ferv*", "interes*", "zeal*"},
    "irritation": {"annoy*", "exasperat*", "grump*", "indign*", "irrita*", "sullen*", "vex*"},
    "jealousy": {"covetous*", "jealous*"},
    "joy": {"ecstat*", "elat*", "euphor*", "exalt*", "exhilar*", "exult*", "flush*", "glee*", "joy*", "jubil*", "overjoyed",	"ravish*",	"rejoic*"},
    "longing": {"crav*", "daydream*", "desir*", "fanta*", "hanker*", "hark*", "homesick*", "long*", "nostalg*", "pin*", "regret*", "wish*", "wistf*", "yearn*"},
    "lust": {"carnal", "lust*", "climax", "ecsta*", "orgas*", "sensu*", "sexual*"},
    "pleasure": {"enjoy*", "delight*", "glow*", "pleas*", "thrill*", "zest*"},
    "pride": {"pride*", "proud*"},
    "relaxation": {"ease*", "calm*", "carefree", "casual", "detach*", "dispassion*", "equanim*", "eventemper*", "laid-back", "peace*", "placid*", "poise*", "relax*", "seren*", "tranquil*", "unruffl*"},
    "relief": {"relie*"},
    "sadness": {"chagrin*", "deject*", "dole*", "gloom*", "glum*", "grie*", "hopeles*", "melancho*", "mourn*", "sad*", "sorrow*", "tear*", "weep*"},
    "shame": {"abash*", "asham*", "crush*", "disgrace*", "embarras*", "humili*", "shame*"},
    "surprise": {"amaze*", "astonish*", "dumbfound*", "startl*", "stunn*", "surpris*", "aback", "thunderstruck", "wonder*"},
    "tension": {"activ*", "agit*", "discomfort*", "distress*", "strain*", "stress*", "tense*"},
    "positive": {"agree*", "excellentm", "fair", "fine", "good", "nice", "positiv*"},
    "negative": {"bad", "disagree*", "lousy", "negativ*", "unpleas*"}
}

def GALC_assign_emotion(word):
    #stem the word
    word = stemmer_snowball.stem(word)

    #iterate over the emotions
    for curr_emotion in galc_lexicon.keys():
        #iterate over the words in the current lexicon
        for curr_word in galc_lexicon[curr_emotion]:
            #
            if curr_word[-1] == "*":
                curr_word = curr_word[:-1] #remove the *

                if word.startswith(curr_word): ## found a match
                    return curr_emotion
            else:
                if word == curr_word:
                    return curr_emotion

    return None

def GALC_lexicon_extraction(sentence):
    #lowercase
    sentence = sentence.lower()

    #remove non alphabetical characters
    sentence = re.sub(r'[^ \w+]', '', sentence)

    #tokenize
    sentence = nltk.word_tokenize(sentence)

    #define the output
    output = {}
    output['Emotions.GALC_unk'] = 0
    for k in galc_lexicon.keys():
        output[f"Emotions.GALC_{k}"] = 0

    #compute the score
    for word in sentence:
        curr_emotion = GALC_assign_emotion(word)

        if curr_emotion is not None:
            output[f"Emotions.GALC_{curr_emotion}"] += 1
        else:
            output['Emotions.GALC_unk'] +=1

    return output

def emotions_extractor(df):
    if "Emotions.Vader_POS" not in df.columns:
        vs_analyzer = SentimentIntensityAnalyzer()
        vs = [vs_analyzer.polarity_scores(x) for x in df.reviewText]

        #
        df["Emotions.Vader_POS"] = [x['pos'] for x in vs]
        df["Emotions.Vader_NEG"] = [x['neg'] for x in vs]
        df["Emotions.Vader_NEU"] = [x['neu'] for x in vs]


    if "Emotions.SentiWordNet_POS" not in df.columns:
        scores = df["reviewText"].apply(lambda x: sentiwordnet(x))
        df["Emotions.SentiWordNet_POS"] = [x[0] for x in scores]
        df["Emotions.SentiWordNet_OBJ"] = [x[1] for x in scores]
        df["Emotions.SentiWordNet_NEG"] = [x[2] for x in scores]

    if "Emotions.OpinionLexicon_POS" not in df.columns:
        scores = df["reviewText"].apply(lambda x: opinion_lexicon(x))
        df["Emotions.OpinionLexicon_POS"] = [x[0] for x in scores]
        df["Emotions.OpinionLexicon_NEG"] = [x[1] for x in scores]

    if 'Emotions.GALC_unk' not in df.columns:
        #extract
        # galc_scores = df['reviewText'].apply(lambda x: GALC_lexicon_extraction(x)).values
        galc_scores = [GALC_lexicon_extraction(x) for x in df['reviewText'].values]

        galc_df = pd.DataFrame(galc_scores)
        galc_df.index = df.index #assign the same indexes

        #concatenate both the dataframe
        df = pd.concat([df, galc_df], axis = 1)

    return df
