import pickle

import re
import nltk

import pickle

import wordlist as wl

# nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt 

# nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

#spell correction
from autocorrect import spell

doNOT = wl.doNOT

stopword = wl.stopword

def process(sms):
    sms = re.sub('[^A-Za-z]', ' ', sms)

    # make words lowercase, because Go and go will be considered as two words
    sms = sms.lower()

    # tokenising
    tokenized_sms = wt(sms)
    print(tokenized_sms)

    # remove stop words and stemming
 
    sms_processed = []
    for word in tokenized_sms:
        if word not in set(stopword):
            if word in set(doNOT):
                word = "not"
            sms_processed.append(spell(stemmer.stem(word)))

    sms_text = " ".join(sms_processed)
    return sms_text

sent = "can i ask if my item with local track number 5465446564546 arrive at warehouse"

with open('./model/matrix.pkl', 'rb') as fid:
    matrix = pickle.load(fid)
    x_test = matrix.transform([process(sent)]).toarray()
    print(x_test)

    with open('./model/classifier.pkl', 'rb') as fid2:
        classifier = pickle.load(fid2)  

        y_pred = classifier.predict(x_test)
        print(y_pred)


