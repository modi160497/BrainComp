from brian2 import *
import clustering  # cluster.py gives back dictionary of word and which cluster it belongs to
import parsecsv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import AgglomerativeClustering

length = 0

def tokenizerTrain():
    tokens = []
    fintokens = []
    global length

    sentences, testsentences, trainsentence = parsecsv.parse()  # trainsentence is 50,000 negative and 50,000 positive sentences for training data

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
        fintokens.append(tokens)  # contains test sentences tokenized

    length = len(fintokens)

    return fintokens


def processwords(index):  # return input rates for words in a given sentence, based on the cluster
    dictcluser = clustering.clusterword()

    trainsentencetokes, length = tokenizerTrain()

    inputrate = [100]

    for i in range(0, len(trainsentencetokes)):
        word = trainsentencetokes[index][i]

        val = dictcluser[word]

        inputrate[i] = (1 + val) * 0.5

    return inputrate


def neuralnet():

    global length

    for i in range(0, length):

        inputrate = processwords(i)

        
