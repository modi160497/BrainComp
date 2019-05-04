import parsecsv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':

    emotions = []
    sentences = []

    sentences = parsecsv.parse()

    print(len(sentences))

    tokens = []

    tokenizer = RegexpTokenizer(r'\w+') # remove puncutation from sentences

    stop_words = set(stopwords.words('english'))

    for i in range(0, len(sentences)):

        data = tokenizer.tokenize(sentences[i][1])
        filtered_sentence = [w for w in data if not w in stop_words] # remove articles and stop words
        words = [word for word in filtered_sentence if word.isalpha()]
        tokens.append(words)

    print(tokens[0])

    sen_w2v = Word2Vec(size= 200, min_count=8)
    sen_w2v.build_vocab(tokens)
    sen_w2v.train(tokens, total_examples=len(tokens), epochs=10)

    print(sen_w2v.most_similar('accident'))




