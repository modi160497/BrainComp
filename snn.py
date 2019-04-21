import parsecsv

if __name__ == '__main__':

    emotions = []
    sentences = []

    emotions,sentences = parsecsv.parse()

    print(emotions)

    print(sentences)



