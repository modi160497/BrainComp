from brian2 import *
import clustering  # cluster.py gives back dictionary of word and which cluster it belongs to
import parsecsv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import AgglomerativeClustering

def tokenizer():
    tokens = []
    fintokens = []

    sentences, testsentences, trainsentence = parsecsv.parse()

    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')

    for i in range(0, len(sentences)):

        tokens = tokenizer.tokenize(trainsentence[i][1])
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
         # remove articles and stop words
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w for w in tokens if not w in stop_words]
        fintokens.append(tokens) #contains test sentences tokenized

def processwords():
    dictcluser = clustering.clusterword()

