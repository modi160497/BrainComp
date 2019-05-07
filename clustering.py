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

def clusterword():

    tokens = []
    fintokens = []
    length = []

    sentences, testsentences, trainsentence = parsecsv.parse()

    #print(len(sentences))

    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')

    for i in range(0, len(sentences)):

        tokens = tokenizer.tokenize(sentences[i][1])
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
         # remove articles and stop words
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [w for w in tokens if not w in stop_words]
        length.append(len(tokens))
        fintokens.append(tokens)


    print(fintokens[0])
    print(len(fintokens)) # max length of the setence used

    #print(tokens[0])
    print(max(length)) # max length of the setence used

    sen_w2v = Word2Vec(size= 200, min_count=10)
    sen_w2v.build_vocab(fintokens)
    sen_w2v.train(fintokens, total_examples=len(fintokens), epochs=10)


    print(sen_w2v.most_similar('sad'))
    print(sen_w2v.most_similar('love'))


    train_data = list()
    position = 0
    word_position = dict()
    train_words = list()
    for idx, key in enumerate(sen_w2v.wv.vocab):
        train_data.append(sen_w2v.wv[key])
        train_words.append(key)
        word_position[key] = position
        position += 1
    #print(len(train_data))
    clustering = AgglomerativeClustering(n_clusters = 1000)
    labels = clustering.fit_predict(train_data)
    #to find the cluster number of a word, look up word_position[word] to get the index, then labels[index] for cluster label
    f = open('clusters.txt', 'a')
    #sad_cluster = labels[word_position['sad']]
    '''
    for i in range(len(labels)):
        f.write('Word: {}, Cluster: {}\n'.format(train_words[i], labels[i]))
    f.close()
    '''
    for i in range(5000):
        for j in range(len(labels)):
            if labels[j] == i:
                f.write('Word: {}, Cluster: {}\n'.format(train_words[j], labels[j]))

    clusters = dict()
    for i in range(len(train_words)):
        word = train_words[i]
        cl = labels[i]
        clusters[word] = cl

    return clusters
