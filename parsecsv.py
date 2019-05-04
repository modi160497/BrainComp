import csv
import unicodedata
import re

def parse():
    csvpath = "py_isear_dataset\isear.csv"

    fd = open(csvpath, "r")

    dataset_reader = csv.reader(fd,delimiter="|",quotechar='"')

    i = 0

    emotions = []
    sentences = []

    for row in dataset_reader:
        if i == 0:
            i = i + 1
            continue
        emotions.append(row[36])
        normal = unicodedata.normalize('NFKD', row[40]).encode('ASCII', 'ignore')
        normal = normal.decode("utf-8")
        sentences.append(normal)

    print(sentences[0])
    sentiment = []


    for i in range(0, len(emotions)):
        if(emotions[i] == "joy"):
            sentiment.append((0, sentences[i].lower()))
        if(emotions[i] == 'sadness' or emotions[i] == 'anger' or emotions[i] == 'digust' or emotions[i]== 'guilt' or emotions[i] == 'fear'):
            sentiment.append((1, sentences[i].lower()))

    return sentiment


