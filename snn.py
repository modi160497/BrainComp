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

if __name__ == '__main__':

    tokens = []
    fintokens = []
    length = []

    sentences, testsentences = parsecsv.parse()

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

    print(tokens[0])
    print(max(length)) # max length of the setence used

    sen_w2v = Word2Vec(size= 200, min_count=10)
    sen_w2v.build_vocab(fintokens)
    sen_w2v.train(fintokens, total_examples=len(fintokens), epochs=10)


    print(sen_w2v.most_similar('sad'))

    #print(sen_w2v.most_similar('accident'))
    train_data = list()
    for idx, key in enumerate(sen_w2v.wv.vocab):
        train_data.append(sen_w2v.wv[key])
    #print(len(train_data))
    clustering = AgglomerativeClustering(n_clusters = 50)
    labels = clustering.fit_predict(train_data)
    print(labels)
