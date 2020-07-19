import numpy as np
import random
import string
import io
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular',quiet=True)

#use only in first time its run
#nltk.download('punkt')
#nltk.download('wordnet')

f=open('lotr_info.txt','r',errors='ignore')

raw = f.read();
#make everything lowercase
raw= raw.lower()



#list of sentences
sent_tolkiens = nltk.sent_tokenize(raw)
#list of words
word_tolkiens = nltk.word_tokenize(raw)

#preprocess raw text
lemmer = nltk.stem.WordNetLemmatizer()

def LemTolkiens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None)for punct in string.punctuation)
def LemNormalize(text):
    return LemTolkiens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#keyword matching

GREETING_INPUTS = ("hello", "hi","hey", "greetings",)
GREETING_RESPONSES = ["hi", "hello", "hi there!", "welcome"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


#generating responses for questions

def response(user_response):
    bot_response=''
    sent_tolkiens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tolkiens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        bot_response=bot_response+"I'm sorry, I don't quite understand"
        return bot_response
    else:
        bot_response = bot_response+sent_tolkiens[idx]
        return bot_response


flag=True
print("BOT: My name is prof.underhil, I will be leading you on this adventure. If you want to leave, type Bye!")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag = False
            print("BOT: You're welcome :)")
        else:
            if(greeting(user_response)!=None):
                print("BOT: "+greeting(user_response))
            else:
                print("BOT: ",end="")
                print(response(user_response))
                sent_tolkiens.remove(user_response)
    else:
        flag=False
        print("BOT: Bye! Take Care")
