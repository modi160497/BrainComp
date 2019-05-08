from brian2 import *
import clustering  # cluster.py gives back dictionary of word and which cluster it belongs to
import parsecsv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
import numpy as np

tokenizer = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import AgglomerativeClustering
import globales.py  # global variables
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pylab import *
import math
import random


def tokenizerTrain():
    tokens = []
    fintokens = []

    sentences, testsentences, trainsentence = parsecsv.parse()  # trainsentence is 50,000 negative and 50,000 positive sentences for training data

    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')

    for i in range(0, len(sentences)):
        tokens = tokenizer.tokenize(trainsentence[i][1])
        sentiment = trainsentence[i][0]
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        # remove articles and stop words
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w for w in tokens if not w in stop_words]
        globales.length.append(len(tokens))
        fintokens.append((tokens, sentiment))  # contains test sentences tokenized

    globales.length = len(fintokens)

    return fintokens


def initialweights():
    positiveweights = np.ones(100)


negativeweights = np.multiply(-1, np.ones(100))
weights = list()
weights.append(positiveweights)
weights.append(negativeweights)
return weights


def clusterwords():
    cluster = dict()
    with open('clusters.txt') as f:
        line = f.getline().split()
    key = line[1][:-1]
    value = line[-1]
    cluster[key] = value

    return cluster


def processwords():
    globales.tokens = tokenizerTrain()
    for i in range(len(globales.tokens)):
        sentence = globales.tokens[i][0]
        sentim = globales.tokens[i][1]
        for word in sentence:
            cl = globales.dictcluster[word]
        if sentim == 1:
            globales.clusterrate[cl] += 1
        elif sentim == 0:
            globales.clusterrate[cl] -= 1
    # clusterweights is a list of size 1000 where each cluster's sum of sentiment


def inputrate(sentence_list):
    firerates = np.zeros(100)
    for i in range(len(sentence_list)):
        word = sentence_list[i]
    cl = globales.dictcluster[word]
    firerates[i] = globales.clusterrate[cl]


return firerates


def simulate_iz(_I, a, b, c, d, check):
    spikes = []

    tmax = 1000
    dt = 0.5

    Iapp = _I
    tr = array([200., 700]) / dt

    T = (int)(ceil(tmax / dt))
    v = zeros(T)
    u = zeros(T)
    v[0] = -70
    u[0] = -14

    for t in arange(T - 1):

        if t > tr[0] and t < tr[1]:
            I = Iapp
        else:
            I = 0

        if v[t] < 35:
            spikes.append(0)
            dv = (0.04 * v[t] + 5) * v[t] + 140 - u[t]
            v[t + 1] = v[t] + (dv + I) * dt
            du = a * (b * v[t] - u[t])
            u[t + 1] = u[t] + dt * du
        else:
            spikes.append(1)
            v[t] = 35
            v[t + 1] = c
            u[t + 1] = u[t] + d

    tvec = arange(0., tmax, dt)
    plot(tvec, v, 'b', label='Voltage trace')
    title('Izhikevich')
    ylabel('Membrane Potential (V)')
    xlabel('Time (msec)')
    if (check):
        show()

    return spikes


def neuralnetTrain():
    outputspikes = []
    outputcurrents = []
    positiveoutputcurr = []
    negativeoutputcurr = []
    outputcurr1 = 0
    outputcurr2 = 0
    spiketrain1 = 0
    spiketrain2 = 0
    answers = []

    for i in range(0, length):  # each sentence in the training set
        tokenwords = globales.tokens[i][0]  # tokenwords: tokens in each sentence
        inputrate = inputrates(tokenwords)  # gives a list of input firing rates for each word in tokenwords
        for j in range(0, len(tokenwords)):
            rate = inputrate[j]
            spiketrain = simulate_iz(rate, 0.02, 0.2, -65, 8, false)
        outputspikes.append(spiketrain)  # append spiketrain for each input

        for k in range(0, len(outputspikes))

            tf = 0

            time_steps = len(outputspikes[k])

            for m in range(0, time_steps):
                if (spiketrain1[m] == 1):
                    current = current + 1
                    # print("increase in current")
                    # print(current1)
                    tf = m * 0.05
                else:
                    # print("decrease in current")
                    current = current * math.exp(-t)

                outputcurrents.append(current)

        for n in range(0, len(outputcurrents)):
            outputcurr1 += outputcurrents[n] * 0.1

            outputcurr2 += outputcurrents[n] * -0.1

        spiketrain1 = simulate_iz(outputcurr1, 0.02, 0.2, -65, 8, false)

        spiketrain2 = simulate_iz(outputcurr2, 0.02, 0.2, -65, 8, false)

        sum1 = sum(spiketrain1)

        sum2 = sum(spiketrain2)

        if (sum1 > sum2):
            print("sentence is positive")
            answers.append(1)
        else:
            print("sentence is negative")
            answers.append(0)


globales.dictcluster = clusterwords()
neuralnetTrain()



